"""A deterministic engine with configurable latency and no model weights.

It exists for two reasons:
  * tests and CI can exercise the full serving stack without downloading a model;
  * benchmarks can isolate *serving-layer* overhead (queueing, threading, HTTP, SSE) from model
    compute, because the model's contribution is known exactly (``ttft_s + n * token_latency_s``).

Its ``time.sleep`` blocks the worker thread just like a real backend's decode loop would.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from .base import Chunk, Engine, GenerationParams


class FakeEngine(Engine):
    name = "fake"

    def __init__(self, ttft_s: float = 0.0, token_latency_s: float = 0.0) -> None:
        self.ttft_s = ttft_s
        self.token_latency_s = token_latency_s

    def load(self) -> None:
        pass

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, params: GenerationParams) -> Iterator[Chunk]:
        if self.ttft_s:
            time.sleep(self.ttft_s)
        produced = ""
        for i in range(params.max_tokens):
            if i and self.token_latency_s:
                time.sleep(self.token_latency_s)
            text = f" tok{i}"
            produced += text
            if any(s in produced for s in params.stop):
                yield Chunk(text="", finish_reason="stop")
                return
            yield Chunk(text=text)
        yield Chunk(text="", finish_reason="length")
