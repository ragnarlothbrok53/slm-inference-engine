"""llama.cpp backend (GGUF models) via ``llama-cpp-python``."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .base import Chunk, Engine, GenerationParams


class LlamaCppEngine(Engine):
    name = "llama_cpp"

    def __init__(
        self, model_path: str, n_ctx: int, n_threads: int | None, n_gpu_layers: int
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._llm: Any = None

    def load(self) -> None:
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
        from llama_cpp import Llama  # imported lazily: optional dependency

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            # Same count for prompt processing. llama-cpp-python otherwise uses every logical
            # core for prefill; on a hybrid laptop CPU that measured slower and much noisier.
            n_threads_batch=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    def count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))

    def generate(self, prompt: str, params: GenerationParams) -> Iterator[Chunk]:
        stream = self._llm.create_completion(
            prompt=prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=list(params.stop) or None,
            seed=params.seed,
            stream=True,
        )
        try:
            for part in stream:
                choice = part["choices"][0]
                yield Chunk(text=choice["text"], finish_reason=choice.get("finish_reason"))
        finally:
            # Closing the llama-cpp generator stops decoding immediately (cancellation path).
            stream.close()

    def close(self) -> None:
        if self._llm is not None:
            self._llm.close()
            self._llm = None
