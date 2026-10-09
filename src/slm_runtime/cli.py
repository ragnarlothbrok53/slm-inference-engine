"""Command-line entry point: ``slm-runtime serve | info``.

CLI flags override ``SLM_*`` environment variables, which override defaults.
"""

from __future__ import annotations

import json
from typing import Any

import typer
import uvicorn

from .config import Settings
from .hardware import detect_hardware
from .logs import configure_logging

app = typer.Typer(help="slm-runtime: serve small language models locally.", no_args_is_help=True)


@app.command()
def serve(
    model: str | None = typer.Option(None, help="Path to model (GGUF file or MLX model dir)."),
    engine: str | None = typer.Option(None, help="llama_cpp | mlx | fake"),
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    replicas: int | None = typer.Option(None, help="Independent model instances."),
    n_threads: int | None = typer.Option(None, help="CPU threads per replica."),
) -> None:
    """Start the HTTP server."""
    overrides: dict[str, Any] = {
        k: v
        for k, v in {
            "model_path": model,
            "engine": engine,
            "host": host,
            "port": port,
            "replicas": replicas,
            "n_threads": n_threads,
        }.items()
        if v is not None
    }
    settings = Settings(**overrides)
    configure_logging(settings.log_level, settings.log_json)

    from .app import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,  # keep our loguru configuration
        timeout_graceful_shutdown=10,
    )


@app.command()
def info() -> None:
    """Print detected hardware and effective configuration."""
    print(
        json.dumps({"hardware": detect_hardware(), "settings": Settings().model_dump()}, indent=2)
    )


if __name__ == "__main__":
    app()
