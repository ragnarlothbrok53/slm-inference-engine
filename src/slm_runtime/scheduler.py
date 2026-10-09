"""Request scheduling: admission control, thread bridging, deadlines and cancellation.

Model
-----
* Each engine replica gets one asyncio *worker* and one dedicated OS thread. A replica serves
  exactly one request at a time (llama.cpp / MLX model objects are not safe for concurrent use).
* Requests wait in a single FIFO queue. Total capacity is ``replicas + max_queue_size``; beyond
  that, ``submit`` fails fast with ``QueueFullError`` (HTTP 503) instead of letting latency grow
  without bound.
* The engine's blocking token loop runs on the replica thread; each token is handed back to the
  event loop with ``call_soon_threadsafe``, so the event loop never blocks on inference.
* A request has one deadline covering queue wait + generation, enforced on the consumer side.
  Timeouts and client disconnects set a ``threading.Event`` that the replica thread checks between
  tokens, so abandoned work stops within one token instead of running to ``max_tokens``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from loguru import logger

from .engines import Engine, FinishReason, GenerationParams
from .metrics import Metrics


class SchedulerError(Exception):
    """Base class for errors that map to a well-defined HTTP response."""


class NotReadyError(SchedulerError):
    pass


class QueueFullError(SchedulerError):
    pass


class DeadlineExceededError(SchedulerError):
    pass


class EngineError(SchedulerError):
    pass


# --- events passed from replica threads to the request's consumer -----------------------------


@dataclass(frozen=True)
class _Started:
    prompt_tokens: int


@dataclass(frozen=True)
class _Token:
    text: str


@dataclass(frozen=True)
class _Done:
    finish_reason: FinishReason


@dataclass(frozen=True)
class _Failed:
    error: Exception


_Event = _Started | _Token | _Done | _Failed


@dataclass
class _Job:
    prompt: str
    params: GenerationParams
    loop: asyncio.AbstractEventLoop
    submitted_at: float  # time.perf_counter(), for measuring durations
    deadline: float  # loop.time() based, for asyncio.timeout_at
    events: asyncio.Queue[_Event] = field(default_factory=asyncio.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)

    def emit(self, event: _Event) -> None:
        """Thread-safe: deliver an event to the consumer on the event loop."""
        try:
            self.loop.call_soon_threadsafe(self.events.put_nowait, event)
        except RuntimeError:  # loop closed during shutdown; nobody is listening
            pass


class Generation:
    """Async iterator over generated text for one request, plus its timing/usage stats."""

    def __init__(self, job: _Job, metrics: Metrics) -> None:
        self._job = job
        self._metrics = metrics
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.finish_reason: FinishReason | None = None
        self.queue_wait_s: float | None = None
        self.ttft_s: float | None = None
        self.duration_s: float | None = None
        self._finished = False

    async def __aiter__(self) -> AsyncIterator[str]:
        job = self._job
        try:
            async with asyncio.timeout_at(job.deadline):
                while True:
                    ev = await job.events.get()
                    if isinstance(ev, _Started):
                        self.prompt_tokens = ev.prompt_tokens
                        self.queue_wait_s = time.perf_counter() - job.submitted_at
                        self._metrics.queue_wait.observe(self.queue_wait_s)
                    elif isinstance(ev, _Token):
                        if self.ttft_s is None:
                            self.ttft_s = time.perf_counter() - job.submitted_at
                            self._metrics.ttft.observe(self.ttft_s)
                        self.completion_tokens += 1
                        yield ev.text
                    elif isinstance(ev, _Done):
                        self.finish_reason = ev.finish_reason
                        self._finish("ok")
                        return
                    else:
                        raise EngineError(str(ev.error)) from ev.error
        except TimeoutError:
            self._finish("timeout")
            raise DeadlineExceededError("request deadline exceeded") from None
        except EngineError:
            self._finish("engine_error")
            raise
        finally:
            self.close()

    def close(self) -> None:
        """Abandon the request if still running (e.g. client disconnected). Idempotent."""
        if not self._finished:
            self._finish("cancelled")

    def _finish(self, outcome: str) -> None:
        self._finished = True
        self._job.cancelled.set()  # no-op if generation already ended; stops it otherwise
        self.duration_s = time.perf_counter() - self._job.submitted_at
        m = self._metrics
        m.requests.labels(outcome=outcome).inc()
        m.generated_tokens.inc(self.completion_tokens)
        m.prompt_tokens.inc(self.prompt_tokens)
        if outcome == "ok":
            m.request_duration.observe(self.duration_s)
            if self.ttft_s is not None and self.completion_tokens > 1:
                decode_s = self.duration_s - self.ttft_s
                if decode_s > 0:
                    m.decode_tokens_per_second.observe((self.completion_tokens - 1) / decode_s)


class Scheduler:
    def __init__(self, engines: list[Engine], max_queue_size: int, metrics: Metrics) -> None:
        if not engines:
            raise ValueError("at least one engine replica is required")
        self._engines = engines
        self._max_queue_size = max_queue_size
        self._metrics = metrics
        self._executors = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"replica-{i}")
            for i in range(len(engines))
        ]
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._running: dict[int, _Job] = {}
        self._pending = 0  # submitted, not yet picked up by a worker
        self._ready = False
        self._stopping = False
        self.load_error: Exception | None = None
        metrics.replicas.set(len(engines))

    # --- lifecycle ---------------------------------------------------------------------------

    async def start(self) -> None:
        """Load all replicas (concurrently, each on its own thread), then start workers."""
        loop = asyncio.get_running_loop()
        try:
            await asyncio.gather(
                *(
                    loop.run_in_executor(ex, eng.load)
                    for ex, eng in zip(self._executors, self._engines, strict=True)
                )
            )
        except Exception as exc:
            self.load_error = exc
            logger.exception("model load failed")
            return
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"worker-{i}")
            for i in range(len(self._engines))
        ]
        self._ready = True
        logger.info("scheduler ready", replicas=len(self._engines))

    async def stop(self) -> None:
        """Reject new work, fail queued jobs, stop in-flight generation, release engines."""
        self._stopping = True
        self._ready = False
        while not self._queue.empty():
            job = self._queue.get_nowait()
            job.emit(_Failed(NotReadyError("server shutting down")))
        for job in self._running.values():
            job.cancelled.set()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        loop = asyncio.get_running_loop()
        for ex, eng in zip(self._executors, self._engines, strict=True):
            await loop.run_in_executor(ex, eng.close)  # waits for the in-flight token to finish
            ex.shutdown(wait=True)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def capacity(self) -> int:
        return len(self._engines) + self._max_queue_size

    @property
    def in_system(self) -> int:
        return self._pending + len(self._running)

    # --- request path ------------------------------------------------------------------------

    def submit(self, prompt: str, params: GenerationParams, timeout_s: float) -> Generation:
        if not self._ready or self._stopping:
            raise NotReadyError("model is not loaded" if not self._stopping else "shutting down")
        if self.in_system >= self.capacity:
            self._metrics.rejected.labels(reason="queue_full").inc()
            raise QueueFullError(f"server at capacity ({self.capacity} requests)")
        loop = asyncio.get_running_loop()
        now = loop.time()
        job = _Job(prompt, params, loop, submitted_at=time.perf_counter(), deadline=now + timeout_s)
        self._pending += 1
        self._queue.put_nowait(job)
        self._update_gauges()
        return Generation(job, self._metrics)

    async def _worker(self, idx: int) -> None:
        loop = asyncio.get_running_loop()
        engine, executor = self._engines[idx], self._executors[idx]
        while True:
            job = await self._queue.get()
            self._pending -= 1
            if job.cancelled.is_set():  # timed out or disconnected while queued: skip the work
                self._update_gauges()
                continue
            self._running[idx] = job
            self._update_gauges()
            try:
                await loop.run_in_executor(executor, _run_job, engine, job)
            finally:
                del self._running[idx]
                self._update_gauges()

    def _update_gauges(self) -> None:
        self._metrics.queue_depth.set(self._pending)
        self._metrics.inflight.set(len(self._running))


def _run_job(engine: Engine, job: _Job) -> None:
    """Runs on a replica thread. Never raises: every outcome is reported as an event."""
    try:
        job.emit(_Started(prompt_tokens=engine.count_tokens(job.prompt)))
        gen = engine.generate(job.prompt, job.params)
        try:
            for chunk in gen:
                if job.cancelled.is_set():
                    return
                if chunk.text:
                    job.emit(_Token(chunk.text))
                if chunk.finish_reason:
                    job.emit(_Done(chunk.finish_reason))
                    return
            job.emit(_Done("stop"))
        finally:
            close = getattr(gen, "close", None)
            if close is not None:
                close()
    except Exception as exc:
        logger.exception("generation failed")
        job.emit(_Failed(exc))
