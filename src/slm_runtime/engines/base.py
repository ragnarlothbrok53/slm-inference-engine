"""The engine contract every backend implements.

Engines are deliberately *synchronous*: llama.cpp and MLX both expose blocking, thread-affine
APIs. The scheduler owns all concurrency — it runs each engine on a dedicated thread and bridges
tokens back to the asyncio event loop. Keeping engines dumb makes new backends easy to add and
keeps all queueing/timeout/cancellation logic in one tested place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int
    temperature: float = 0.8
    top_p: float = 0.95
    stop: tuple[str, ...] = field(default_factory=tuple)
    seed: int | None = None


@dataclass(frozen=True)
class Chunk:
    """One decoded token's text. The last chunk carries ``finish_reason`` (text may be empty)."""

    text: str
    finish_reason: FinishReason | None = None


class Engine(ABC):
    name: str

    @abstractmethod
    def load(self) -> None:
        """Load weights. Slow; called once on the engine's worker thread before serving."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Number of tokens ``text`` encodes to (used for usage accounting)."""

    @abstractmethod
    def generate(self, prompt: str, params: GenerationParams) -> Iterator[Chunk]:
        """Lazily yield chunks. Callers stop generation early by closing the iterator."""

    def close(self) -> None:  # noqa: B027 - optional hook, default no-op
        """Release resources (weights, file handles)."""
