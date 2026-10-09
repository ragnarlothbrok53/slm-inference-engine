"""MLX backend for Apple Silicon via ``mlx-lm``.

Not exercised in CI (requires macOS on arm64). Stop sequences are matched on the accumulated
output; a stop string that spans several tokens may have its leading tokens already streamed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .base import Chunk, Engine, GenerationParams


class MLXEngine(Engine):
    name = "mlx"

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        from mlx_lm import load  # imported lazily: optional, macOS-only dependency

        self._model, self._tokenizer = load(self.model_path)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def generate(self, prompt: str, params: GenerationParams) -> Iterator[Chunk]:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        if params.seed is not None:
            mx.random.seed(params.seed)
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)
        produced = ""
        for resp in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=params.max_tokens,
            sampler=sampler,
        ):
            produced += resp.text
            for s in params.stop:
                idx = produced.find(s)
                if idx != -1:
                    keep = len(resp.text) - (len(produced) - idx)
                    yield Chunk(text=resp.text[: max(keep, 0)], finish_reason="stop")
                    return
            yield Chunk(text=resp.text, finish_reason=resp.finish_reason)
