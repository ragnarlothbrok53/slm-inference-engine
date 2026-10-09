"""FastAPI application: routes, error mapping, request-ID/access-log middleware, lifecycle."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__
from .config import Settings
from .engines import Engine, FinishReason, GenerationParams, create_engine
from .metrics import Metrics
from .scheduler import (
    DeadlineExceededError,
    EngineError,
    Generation,
    NotReadyError,
    QueueFullError,
    Scheduler,
    SchedulerError,
)
from .schemas import CompletionChoice, CompletionRequest, CompletionResponse, Usage

# Scheduler error -> (HTTP status, OpenAI-style error type)
_ERROR_MAP: dict[type[SchedulerError], tuple[int, str]] = {
    NotReadyError: (503, "service_unavailable"),
    QueueFullError: (503, "overloaded"),
    DeadlineExceededError: (504, "timeout"),
    EngineError: (500, "engine_error"),
}


def _error_body(message: str, type_: str) -> dict[str, dict[str, str]]:
    return {"error": {"message": message, "type": type_}}


class RequestContextMiddleware:
    """Assigns/propagates ``X-Request-ID`` and emits one structured access-log line per request.

    Pure ASGI (not BaseHTTPMiddleware) so the log line is written when a streamed response has
    actually finished, giving true end-to-end duration.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        request_id = headers.get(b"x-request-id", b"").decode() or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        with logger.contextualize(request_id=request_id):
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                if scope["path"] not in ("/healthz", "/readyz", "/metrics"):  # probe/scrape noise
                    logger.info(
                        "{method} {path} {status}",
                        method=scope["method"],
                        path=scope["path"],
                        status=status,
                        duration_ms=round((time.perf_counter() - start) * 1000, 1),
                    )


def create_app(settings: Settings | None = None, engines: list[Engine] | None = None) -> FastAPI:
    """Build the app. ``engines`` lets tests inject engines; by default they come from settings."""
    settings = settings or Settings()
    if engines is None:
        engines = [create_engine(settings) for _ in range(settings.replicas)]
    metrics = Metrics()
    scheduler = Scheduler(engines, settings.max_queue_size, metrics)
    model_id = Path(settings.model_path).name if settings.engine != "fake" else "fake"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load in the background: the process answers /healthz immediately and /readyz flips to
        # 200 once weights are loaded, which is what orchestrators need for slow model loads.
        load_task = asyncio.create_task(scheduler.start())
        try:
            yield
        finally:
            load_task.cancel()
            await asyncio.gather(load_task, return_exceptions=True)
            await scheduler.stop()

    app = FastAPI(title="slm-runtime", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.state.scheduler = scheduler
    app.state.metrics = metrics
    app.state.settings = settings

    @app.exception_handler(SchedulerError)
    async def _scheduler_error(_: Request, exc: SchedulerError) -> JSONResponse:
        status, type_ = _ERROR_MAP.get(type(exc), (500, "internal_error"))
        headers = {"Retry-After": "1"} if status == 503 else None
        return JSONResponse(_error_body(str(exc), type_), status_code=status, headers=headers)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            _error_body(str(exc.detail), "invalid_request_error"), status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        msg = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()
        )
        return JSONResponse(_error_body(msg, "invalid_request_error"), status_code=400)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: the process and event loop are responsive."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """Readiness: weights are loaded and the scheduler accepts work."""
        if scheduler.ready:
            body = {"status": "ready", "replicas": len(engines), "in_system": scheduler.in_system}
            return JSONResponse(body)
        if scheduler.load_error is not None:
            body = {"status": "failed", "error": str(scheduler.load_error)}
        else:
            body = {"status": "loading"}
        return JSONResponse(body, status_code=503)

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/completions", response_model=None)
    async def completions(req: CompletionRequest) -> CompletionResponse | StreamingResponse:
        if req.max_tokens > settings.max_tokens_limit:
            raise HTTPException(400, f"max_tokens must be <= {settings.max_tokens_limit}")
        if len(req.prompt) > settings.max_prompt_chars:
            raise HTTPException(400, f"prompt exceeds {settings.max_prompt_chars} characters")
        params = GenerationParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=tuple(req.stop or ()),
            seed=req.seed,
        )
        # Admission happens before any response bytes are sent, so overload/not-ready errors
        # get proper status codes even for streaming requests.
        gen = scheduler.submit(req.prompt, params, settings.request_timeout_s)
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _sse(gen, completion_id, created, model_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            text = "".join([piece async for piece in gen])
        finally:
            gen.close()
        _log_generation(gen)
        return CompletionResponse(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[CompletionChoice(text=text, finish_reason=gen.finish_reason)],
            usage=_usage(gen),
        )

    return app


def _usage(gen: Generation) -> Usage:
    return Usage(
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
        total_tokens=gen.prompt_tokens + gen.completion_tokens,
    )


def _log_generation(gen: Generation) -> None:
    def ms(v: float | None) -> float | None:
        return None if v is None else round(v * 1000, 1)

    logger.info(
        "generation finished",
        finish_reason=gen.finish_reason,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
        queue_wait_ms=ms(gen.queue_wait_s),
        ttft_ms=ms(gen.ttft_s),
        duration_ms=ms(gen.duration_s),
    )


async def _sse(gen: Generation, cid: str, created: int, model: str) -> AsyncIterator[str]:
    def event(text: str, finish_reason: FinishReason | None, usage: Usage | None = None) -> str:
        chunk = CompletionResponse(
            id=cid,
            created=created,
            model=model,
            choices=[CompletionChoice(text=text, finish_reason=finish_reason)],
            usage=usage,
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    try:
        async for piece in gen:
            yield event(piece, None)
        yield event("", gen.finish_reason, _usage(gen))
        _log_generation(gen)
    except SchedulerError as exc:
        # Headers are already sent; report the failure in-band, then end the stream.
        _, type_ = _ERROR_MAP.get(type(exc), (500, "internal_error"))
        yield f"data: {json.dumps(_error_body(str(exc), type_))}\n\n"
    finally:
        gen.close()  # runs on client disconnect too: stops generation on the replica thread
    yield "data: [DONE]\n\n"
