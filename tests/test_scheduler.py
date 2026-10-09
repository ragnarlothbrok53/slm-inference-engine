"""Scheduler behaviour: ordering, admission control, deadlines, cancellation, failure isolation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from helpers import BrokenLoadEngine, FailingEngine, InstrumentedEngine

from slm_runtime.engines import Engine, GenerationParams
from slm_runtime.metrics import Metrics
from slm_runtime.scheduler import (
    DeadlineExceededError,
    EngineError,
    Generation,
    NotReadyError,
    QueueFullError,
    Scheduler,
)

pytestmark = pytest.mark.anyio


@asynccontextmanager
async def running(engines: list[Engine], max_queue_size: int = 8) -> AsyncIterator[Scheduler]:
    sched = Scheduler(engines, max_queue_size, Metrics())
    await sched.start()
    try:
        yield sched
    finally:
        await sched.stop()


async def collect(gen: Generation) -> str:
    return "".join([t async for t in gen])


def p(max_tokens: int = 5, stop: tuple[str, ...] = ()) -> GenerationParams:
    return GenerationParams(max_tokens=max_tokens, stop=stop)


async def test_generates_tokens_and_reports_usage() -> None:
    async with running([InstrumentedEngine()]) as s:
        gen = s.submit("one two three", p(max_tokens=4), timeout_s=5)
        assert await collect(gen) == " tok0 tok1 tok2 tok3"
        assert gen.finish_reason == "length"
        assert (gen.prompt_tokens, gen.completion_tokens) == (3, 4)
        assert gen.ttft_s is not None and gen.queue_wait_s is not None


async def test_stop_sequence_ends_generation() -> None:
    async with running([InstrumentedEngine()]) as s:
        gen = s.submit("x", p(max_tokens=50, stop=("tok2",)), timeout_s=5)
        assert await collect(gen) == " tok0 tok1"
        assert gen.finish_reason == "stop"


async def test_single_replica_never_runs_requests_concurrently() -> None:
    engine = InstrumentedEngine(token_latency_s=0.005)
    async with running([engine]) as s:
        gens = [s.submit(f"r{i}", p(), timeout_s=5) for i in range(4)]
        await asyncio.gather(*(collect(g) for g in gens))
    assert engine.max_active == 1
    assert engine.prompts == ["r0", "r1", "r2", "r3"]  # FIFO


async def test_replicas_serve_requests_in_parallel() -> None:
    engines = [InstrumentedEngine(token_latency_s=0.02) for _ in range(2)]
    async with running(engines) as s:  # type: ignore[arg-type]
        gens = [s.submit(f"r{i}", p(), timeout_s=5) for i in range(2)]
        await asyncio.gather(*(collect(g) for g in gens))
    assert [len(e.prompts) for e in engines] == [1, 1]


async def test_rejects_when_queue_is_full() -> None:
    metrics = Metrics()
    sched = Scheduler([InstrumentedEngine(token_latency_s=0.05)], max_queue_size=1, metrics=metrics)
    await sched.start()
    try:
        first = sched.submit("a", p(), timeout_s=5)  # occupies the replica
        second = sched.submit("b", p(), timeout_s=5)  # waits in the queue
        with pytest.raises(QueueFullError):
            sched.submit("c", p(), timeout_s=5)
        assert metrics.rejected.labels(reason="queue_full")._value.get() == 1
        await asyncio.gather(collect(first), collect(second))
        # Capacity frees up once work drains.
        await collect(sched.submit("d", p(max_tokens=1), timeout_s=5))
    finally:
        await sched.stop()


async def test_zero_queue_still_admits_up_to_replica_count() -> None:
    async with running([InstrumentedEngine(token_latency_s=0.05)], max_queue_size=0) as s:
        gen = s.submit("a", p(), timeout_s=5)
        with pytest.raises(QueueFullError):
            s.submit("b", p(), timeout_s=5)
        await collect(gen)


async def test_deadline_during_generation_stops_the_engine() -> None:
    engine = InstrumentedEngine(token_latency_s=0.02)
    async with running([engine]) as s:
        gen = s.submit("slow", p(max_tokens=1000), timeout_s=0.1)
        with pytest.raises(DeadlineExceededError):
            await collect(gen)
        # The replica is released within ~one token, so a follow-up request is served promptly
        # (without cancellation it would be stuck behind ~20s of abandoned generation).
        assert await collect(s.submit("next", p(max_tokens=1), timeout_s=0.5)) == " tok0"
        assert engine.tokens_emitted < 20


async def test_deadline_while_queued_skips_the_work() -> None:
    engine = InstrumentedEngine(token_latency_s=0.02)
    async with running([engine]) as s:
        blocker = s.submit("blocker", p(max_tokens=10), timeout_s=5)
        queued = s.submit("queued", p(), timeout_s=0.05)
        with pytest.raises(DeadlineExceededError):
            await collect(queued)
        await collect(blocker)
        await asyncio.sleep(0.05)
    assert engine.prompts == ["blocker"]  # the expired request never reached the model


async def test_closing_a_generation_cancels_backend_work() -> None:
    engine = InstrumentedEngine(token_latency_s=0.01)
    async with running([engine]) as s:
        gen = s.submit("x", p(max_tokens=1000), timeout_s=5)
        async for _ in gen:
            break  # consumer walks away after the first token
        gen.close()
        await asyncio.sleep(0.1)
        assert engine.tokens_emitted < 15
        # The replica is free again for the next request.
        assert await collect(s.submit("y", p(max_tokens=2), timeout_s=5)) == " tok0 tok1"


async def test_engine_failure_is_isolated_to_one_request() -> None:
    async with running([FailingEngine(fail_after=2)]) as s:
        with pytest.raises(EngineError, match="simulated backend crash"):
            await collect(s.submit("boom", p(), timeout_s=5))
        assert await collect(s.submit("fine", p(max_tokens=2), timeout_s=5)) == " tok0 tok1"


async def test_submit_before_ready_is_rejected() -> None:
    sched = Scheduler([InstrumentedEngine()], 4, Metrics())
    with pytest.raises(NotReadyError):
        sched.submit("x", p(), timeout_s=5)


async def test_load_failure_is_recorded_not_raised() -> None:
    sched = Scheduler([BrokenLoadEngine()], 4, Metrics())
    await sched.start()
    assert not sched.ready
    assert isinstance(sched.load_error, FileNotFoundError)
    with pytest.raises(NotReadyError):
        sched.submit("x", p(), timeout_s=5)


async def test_stop_fails_queued_requests_and_closes_engines() -> None:
    engine = InstrumentedEngine(token_latency_s=0.02)
    sched = Scheduler([engine], 4, Metrics())
    await sched.start()
    running_gen = sched.submit("a", p(max_tokens=1000), timeout_s=5)
    queued_gen = sched.submit("b", p(), timeout_s=5)
    await asyncio.sleep(0.05)
    await sched.stop()
    # Queued work fails with a retryable "not ready" error (HTTP 503), not an engine error.
    with pytest.raises(NotReadyError, match="shutting down"):
        await collect(queued_gen)
    running_gen.close()
    assert engine.closed
    assert engine.prompts == ["a"]
    with pytest.raises(NotReadyError):
        sched.submit("c", p(), timeout_s=5)


async def test_metrics_track_outcomes_and_tokens() -> None:
    metrics = Metrics()
    sched = Scheduler([InstrumentedEngine()], 4, metrics)
    await sched.start()
    try:
        await collect(sched.submit("a b", p(max_tokens=3), timeout_s=5))
        g = sched.submit("x", p(max_tokens=100), timeout_s=5)
        async for _ in g:
            break
        g.close()
    finally:
        await sched.stop()
    assert metrics.requests.labels(outcome="ok")._value.get() == 1
    assert metrics.requests.labels(outcome="cancelled")._value.get() == 1
    assert metrics.prompt_tokens._value.get() == 3
    assert metrics.generated_tokens._value.get() >= 4
