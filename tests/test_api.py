"""HTTP contract: status codes, response shapes, SSE framing, probes, metrics, request IDs."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from helpers import BrokenLoadEngine, FailingEngine, InstrumentedEngine, serve

pytestmark = pytest.mark.anyio


def sse_events(body: str) -> list[str]:
    return [line.removeprefix("data: ") for line in body.split("\n\n") if line.startswith("data: ")]


async def test_completion_response_shape() -> None:
    async with serve([InstrumentedEngine()]) as (client, _):
        r = await client.post("/v1/completions", json={"prompt": "a b c", "max_tokens": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    assert body["choices"] == [{"index": 0, "text": " tok0 tok1 tok2", "finish_reason": "length"}]
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}


async def test_max_tokens_from_request_is_honoured() -> None:
    # Regression: the original server ignored max_tokens and always generated 128.
    async with serve([InstrumentedEngine()]) as (client, _):
        r = await client.post("/v1/completions", json={"prompt": "x", "max_tokens": 7})
    assert r.json()["usage"]["completion_tokens"] == 7


async def test_stop_accepts_string_or_list() -> None:
    async with serve([InstrumentedEngine()]) as (client, _):
        r1 = await client.post("/v1/completions", json={"prompt": "x", "stop": "tok1"})
        r2 = await client.post("/v1/completions", json={"prompt": "x", "stop": ["tok1"]})
    for r in (r1, r2):
        assert r.json()["choices"][0] == {"index": 0, "text": " tok0", "finish_reason": "stop"}


async def test_streaming_sse_framing() -> None:
    async with serve([InstrumentedEngine()]) as (client, _):
        r = await client.post(
            "/v1/completions", json={"prompt": "x", "max_tokens": 3, "stream": True}
        )
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert "".join(c["choices"][0]["text"] for c in chunks) == " tok0 tok1 tok2"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 3
    assert len({c["id"] for c in chunks}) == 1


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"prompt": ""}, "prompt"),
        ({"prompt": "x", "max_tokens": 0}, "max_tokens"),
        ({"prompt": "x", "temperature": 3}, "temperature"),
        ({"prompt": "x", "stop": ["a", "b", "c", "d", "e"]}, "stop"),
        ({"prompt": "x", "max_tokens": 10_000}, "max_tokens"),  # above SLM_MAX_TOKENS_LIMIT
        ({"prompt": "x" * 200}, "prompt"),  # above SLM_MAX_PROMPT_CHARS (set to 100 below)
    ],
)
async def test_invalid_requests_get_400_with_error_envelope(
    payload: dict[str, object], field: str
) -> None:
    async with serve([InstrumentedEngine()], max_prompt_chars=100) as (client, _):
        r = await client.post("/v1/completions", json=payload)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert field in r.json()["error"]["message"]


async def test_overload_returns_503_with_retry_after() -> None:
    engine = InstrumentedEngine(token_latency_s=0.05)
    async with serve([engine], max_queue_size=0) as (client, _):
        busy = asyncio.create_task(
            client.post("/v1/completions", json={"prompt": "x", "max_tokens": 10})
        )
        await asyncio.sleep(0.05)
        r = await client.post("/v1/completions", json={"prompt": "y", "stream": True})
        await busy
    assert r.status_code == 503  # rejected before streaming began, so a real status code
    assert r.headers["retry-after"] == "1"
    assert r.json()["error"]["type"] == "overloaded"


async def test_timeout_returns_504() -> None:
    async with serve([InstrumentedEngine(token_latency_s=0.05)], request_timeout_s=0.1) as (c, _):
        r = await c.post("/v1/completions", json={"prompt": "x", "max_tokens": 500})
    assert r.status_code == 504
    assert r.json()["error"]["type"] == "timeout"


async def test_engine_error_returns_500_and_server_keeps_serving() -> None:
    async with serve([FailingEngine()]) as (client, _):
        bad = await client.post("/v1/completions", json={"prompt": "boom"})
        good = await client.post("/v1/completions", json={"prompt": "ok", "max_tokens": 1})
    assert bad.status_code == 500
    assert bad.json()["error"]["type"] == "engine_error"
    assert good.status_code == 200


async def test_engine_error_mid_stream_is_reported_in_band() -> None:
    async with serve([FailingEngine(fail_after=2)]) as (client, _):
        r = await client.post("/v1/completions", json={"prompt": "boom", "stream": True})
    events = sse_events(r.text)
    assert r.status_code == 200  # headers were already sent when the failure happened
    assert json.loads(events[-2])["error"]["type"] == "engine_error"
    assert events[-1] == "[DONE]"


async def test_readiness_reflects_model_loading() -> None:
    engine = InstrumentedEngine(load_delay_s=0.3)
    async with serve([engine], wait_until_ready=False) as (client, _):
        assert (await client.get("/healthz")).status_code == 200  # alive while loading
        loading = await client.get("/readyz")
        assert (loading.status_code, loading.json()["status"]) == (503, "loading")
        r = await client.post("/v1/completions", json={"prompt": "x"})
        assert r.status_code == 503
        await asyncio.sleep(0.4)
        assert (await client.get("/readyz")).status_code == 200


async def test_readiness_reports_load_failure() -> None:
    async with serve([BrokenLoadEngine()], wait_until_ready=False) as (client, _):
        await asyncio.sleep(0.05)
        r = await client.get("/readyz")
    assert r.status_code == 503
    assert r.json() == {"status": "failed", "error": "model.gguf not found"}


async def test_event_loop_stays_responsive_during_generation() -> None:
    # Regression: the original engine called blocking llama.cpp inside `async def`, freezing the
    # event loop (and every other request, including health checks) for the whole generation.
    engine = InstrumentedEngine(token_latency_s=0.02)
    async with serve([engine]) as (client, _):
        gen = asyncio.create_task(
            client.post("/v1/completions", json={"prompt": "x", "max_tokens": 50})
        )  # ~1s of blocking "inference"
        await asyncio.sleep(0.1)
        t0 = time.perf_counter()
        health = await client.get("/healthz")
        health_latency = time.perf_counter() - t0
        assert not gen.done()
        await gen
    assert health.status_code == 200
    assert health_latency < 0.2


async def test_request_id_is_propagated_or_generated() -> None:
    async with serve([InstrumentedEngine()]) as (client, _):
        given = await client.get("/healthz", headers={"x-request-id": "abc123"})
        generated = await client.get("/healthz")
    assert given.headers["x-request-id"] == "abc123"
    assert len(generated.headers["x-request-id"]) == 16


@pytest.mark.parametrize("bad", ["a" * 65, "id with spaces", "id;level=ERROR", ""])
async def test_unsafe_request_ids_are_replaced(bad: str) -> None:
    # Client-supplied IDs end up in logs; anything outside a short safe charset is replaced.
    async with serve([InstrumentedEngine()]) as (client, _):
        r = await client.get("/healthz", headers={"x-request-id": bad.encode("utf-8")})
    assert r.headers["x-request-id"] != bad
    assert len(r.headers["x-request-id"]) == 16


async def test_metrics_endpoint_exposes_inference_metrics() -> None:
    async with serve([InstrumentedEngine()]) as (client, _):
        await client.post("/v1/completions", json={"prompt": "a b", "max_tokens": 4})
        text = (await client.get("/metrics")).text
    assert 'slm_requests_total{outcome="ok"} 1.0' in text
    assert "slm_generated_tokens_total 4.0" in text
    assert "slm_time_to_first_token_seconds_count 1.0" in text
    assert "slm_queue_depth 0.0" in text
    assert "slm_replicas 1.0" in text


async def test_models_endpoint() -> None:
    async with serve([InstrumentedEngine()]) as (client, _):
        r = await client.get("/v1/models")
    assert r.json()["data"][0]["id"] == "fake"


async def test_unready_server_rejects_with_503_not_crash() -> None:
    async with serve([BrokenLoadEngine()], wait_until_ready=False) as (client, _):
        await asyncio.sleep(0.05)
        r = await client.post("/v1/completions", json={"prompt": "x"})
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "service_unavailable"


def test_unknown_engine_fails_at_startup() -> None:
    # Regression: an unknown ENGINE previously left EngineManager.engine unset (AttributeError
    # on the first request). Now it is rejected by config validation before the server starts.
    from pydantic import ValidationError

    from slm_runtime.config import Settings

    with pytest.raises(ValidationError):
        Settings(engine="tensorrt")  # type: ignore[arg-type]
