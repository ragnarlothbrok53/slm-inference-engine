"""Test doubles and helpers shared across test modules."""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from slm_runtime.app import create_app
from slm_runtime.config import Settings
from slm_runtime.engines import Chunk, Engine, GenerationParams
from slm_runtime.engines.fake import FakeEngine


class InstrumentedEngine(FakeEngine):
    """FakeEngine that records what the scheduler asked of it, for assertions."""

    def __init__(self, token_latency_s: float = 0.0, load_delay_s: float = 0.0) -> None:
        super().__init__(ttft_s=0.0, token_latency_s=token_latency_s)
        self.load_delay_s = load_delay_s
        self.prompts: list[str] = []
        self.tokens_emitted = 0
        self.closed = False
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def load(self) -> None:
        time.sleep(self.load_delay_s)

    def generate(self, prompt: str, params: GenerationParams) -> Iterator[Chunk]:
        with self._lock:
            self.prompts.append(prompt)
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            for chunk in super().generate(prompt, params):
                if chunk.text:
                    self.tokens_emitted += 1
                yield chunk
        finally:
            with self._lock:
                self._active -= 1

    def close(self) -> None:
        self.closed = True


class FailingEngine(FakeEngine):
    """Raises after ``fail_after`` tokens when the prompt contains 'boom'."""

    def __init__(self, fail_after: int = 2) -> None:
        super().__init__()
        self.fail_after = fail_after

    def generate(self, prompt: str, params: GenerationParams) -> Iterator[Chunk]:
        for i, chunk in enumerate(super().generate(prompt, params)):
            if "boom" in prompt and i == self.fail_after:
                raise RuntimeError("simulated backend crash")
            yield chunk


class BrokenLoadEngine(FakeEngine):
    def load(self) -> None:
        raise FileNotFoundError("model.gguf not found")


async def wait_ready(client: httpx.AsyncClient, within_s: float = 5.0) -> None:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        if (await client.get("/readyz")).status_code == 200:
            return
        await _sleep(0.01)
    raise AssertionError("server never became ready")


async def _sleep(s: float) -> None:
    import anyio

    await anyio.sleep(s)


@asynccontextmanager
async def serve(
    engines: list[Engine], ready: bool = True, **settings_overrides: object
) -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI]]:
    """Run the app in-process (with lifespan) and yield an HTTP client bound to it."""
    settings = Settings(engine="fake", **settings_overrides)  # type: ignore[arg-type]
    app = create_app(settings, engines=engines)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            if ready:
                await wait_ready(client)
            yield client, app
