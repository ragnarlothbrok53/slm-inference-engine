"""Engine factory. Backends are imported lazily so optional dependencies stay optional."""

from __future__ import annotations

from ..config import Settings
from .base import Chunk, Engine, FinishReason, GenerationParams

__all__ = ["Chunk", "Engine", "FinishReason", "GenerationParams", "create_engine"]


def create_engine(settings: Settings) -> Engine:
    if settings.engine == "llama_cpp":
        from .llama_cpp import LlamaCppEngine

        return LlamaCppEngine(
            settings.model_path, settings.n_ctx, settings.n_threads, settings.n_gpu_layers
        )
    if settings.engine == "mlx":
        from .mlx import MLXEngine

        return MLXEngine(settings.model_path)
    if settings.engine == "fake":
        from .fake import FakeEngine

        return FakeEngine(settings.fake_ttft_s, settings.fake_token_latency_s)
    raise ValueError(f"unknown engine: {settings.engine!r}")
