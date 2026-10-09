"""HTTP request/response schemas, a subset of the OpenAI ``/v1/completions`` API.

Following that shape means existing OpenAI clients and load-testing tools work unmodified.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CompletionRequest(BaseModel):
    model: str | None = Field(None, description="Ignored; the server serves one model.")
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(128, ge=1)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    stop: list[str] | None = Field(None, max_length=4)
    seed: int | None = None
    stream: bool = False

    @field_validator("stop", mode="before")
    @classmethod
    def _coerce_stop(cls, v: object) -> object:
        return [v] if isinstance(v, str) else v


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Literal["stop", "length"] | None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None
