"""End-to-end tests over a real socket (uvicorn), plus an opt-in test against a real GGUF model."""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from helpers import InstrumentedEngine

from slm_runtime.app import create_app
from slm_runtime.config import Settings
from slm_runtime.engines import Engine


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def live_server(settings: Settings, engines: list[Engine] | None = None) -> Iterator[str]:
    port = _free_port()
    config = uvicorn.Config(
        create_app(settings, engines=engines), host="127.0.0.1", port=port, log_config=None
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base}/readyz").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.05)
        else:
            raise AssertionError("server did not become ready")
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_client_disconnect_mid_stream_stops_generation() -> None:
    engine = InstrumentedEngine(token_latency_s=0.01)
    with live_server(Settings(engine="fake"), engines=[engine]) as base:
        payload = {"prompt": "x", "max_tokens": 1000, "stream": True}  # ~10s if not cancelled
        with httpx.stream("POST", f"{base}/v1/completions", json=payload) as r:
            lines = r.iter_lines()
            assert next(lines).startswith("data: ")
        # Connection closed. The server should notice, cancel, and free the replica.
        t0 = time.monotonic()
        follow_up = httpx.post(
            f"{base}/v1/completions", json={"prompt": "y", "max_tokens": 1}, timeout=5
        )
        assert follow_up.status_code == 200
        assert time.monotonic() - t0 < 2
        metrics = httpx.get(f"{base}/metrics").text
    assert engine.tokens_emitted < 500
    assert 'slm_requests_total{outcome="cancelled"} 1.0' in metrics


MODEL = os.environ.get("SLM_TEST_MODEL_PATH", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")


@pytest.mark.skipif(not Path(MODEL).is_file(), reason=f"real model not present at {MODEL}")
def test_real_llama_cpp_model_end_to_end() -> None:
    pytest.importorskip("llama_cpp")
    settings = Settings(engine="llama_cpp", model_path=MODEL, n_ctx=1024)
    prompt = (
        "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    body = {"prompt": prompt, "max_tokens": 32, "temperature": 0, "stop": ["<|im_end|>"]}
    with live_server(settings) as base:
        r = httpx.post(f"{base}/v1/completions", json=body, timeout=60)
        streamed = httpx.post(f"{base}/v1/completions", json=body | {"stream": True}, timeout=60)
    out = r.json()
    assert r.status_code == 200
    assert "Paris" in out["choices"][0]["text"]
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["prompt_tokens"] > 10
    # Greedy decoding: streaming and non-streaming must produce identical text.
    import json

    events = [ln[6:] for ln in streamed.text.split("\n\n") if ln.startswith("data: {")]
    text = "".join(json.loads(e)["choices"][0]["text"] for e in events)
    assert text == out["choices"][0]["text"]
