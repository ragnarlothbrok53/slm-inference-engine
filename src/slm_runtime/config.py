"""Runtime configuration, loaded from environment variables (prefix ``SLM_``) or a ``.env`` file."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EngineName = Literal["llama_cpp", "mlx", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SLM_", env_file=".env", extra="ignore")

    # --- model / engine ---
    engine: EngineName = "llama_cpp"
    model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    n_ctx: int = Field(4096, ge=128, description="Context window (llama.cpp).")
    n_threads: int | None = Field(
        None, ge=1, description="CPU threads per replica (decode and prefill); None = auto."
    )
    n_gpu_layers: int = Field(0, description="Layers to offload to GPU (llama.cpp); -1 = all.")
    replicas: int = Field(
        1, ge=1, le=16, description="Independent model instances, each served by its own worker."
    )

    # --- scheduling / admission control ---
    max_queue_size: int = Field(
        32, ge=0, description="Requests allowed to wait for a free replica before 503."
    )
    request_timeout_s: float = Field(
        120.0, gt=0, description="Deadline covering queue wait + generation."
    )

    # --- request limits ---
    max_tokens_limit: int = Field(1024, ge=1, description="Upper bound on max_tokens per request.")
    max_prompt_chars: int = Field(32_000, ge=1)

    # --- fake engine (tests / serving-overhead benchmarks) ---
    fake_ttft_s: float = Field(0.0, ge=0, description="Simulated prefill latency.")
    fake_token_latency_s: float = Field(
        0.0, ge=0, description="Simulated per-token decode latency."
    )

    # --- server / observability ---
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = False
