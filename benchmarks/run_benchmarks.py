"""Run named benchmark scenarios end-to-end and regenerate the results report.

For each scenario this script: starts ``slm-runtime serve`` as a subprocess with a fixed config,
waits for /readyz, samples the server process's RSS and CPU while ``loadgen`` drives load, stops
the server, and writes ``benchmarks/results/<scenario>/{summary.json,summary.csv,requests.csv}``.
``benchmarks/results/README.md`` is then regenerated from every summary.json present.

    uv run python benchmarks/run_benchmarks.py --list
    uv run python benchmarks/run_benchmarks.py overhead queueing           # no model needed
    uv run python benchmarks/run_benchmarks.py llama-1x llama-2x llama-long-prompt
    uv run python benchmarks/run_benchmarks.py --report-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

sys.path.insert(0, str(Path(__file__).parent))
import loadgen

from slm_runtime.hardware import detect_hardware

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
MODEL = os.environ.get("SLM_BENCH_MODEL", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")

# ~1.1k-token prompt for measuring prefill cost (TTFT) separately from decode.
_LONG_CONTEXT = " ".join(
    f"Record {i}: the sensor at station {i % 17} reported a reading of {(i * 37) % 101} units."
    for i in range(60)
)
LONG_PROMPT = (
    f"<|im_start|>user\n{_LONG_CONTEXT}\nWhich station reported the highest reading?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@dataclass
class Scenario:
    description: str
    server_env: dict[str, str]
    concurrency: list[int]
    requests: int
    max_tokens: int
    prompt: str = loadgen.DEFAULT_PROMPT
    needs_model: bool = False
    expected: str = ""  # what a correct system should show; checked by reading the numbers
    extra: dict[str, Any] = field(default_factory=dict)


SCENARIOS: dict[str, Scenario] = {
    "overhead": Scenario(
        description="Serving-stack overhead: fake engine with zero model latency, so every "
        "millisecond measured is HTTP + SSE + scheduling + thread hand-off.",
        server_env={"SLM_ENGINE": "fake", "SLM_MAX_QUEUE_SIZE": "256"},
        concurrency=[1, 4, 16, 64],
        requests=400,
        max_tokens=64,
    ),
    "queueing": Scenario(
        description="Scheduler correctness under load: fake engine with a known service time "
        "(50 ms prefill + 31 x 10 ms decode = 360 ms/request) and 1 replica.",
        server_env={
            "SLM_ENGINE": "fake",
            "SLM_FAKE_TTFT_S": "0.05",
            "SLM_FAKE_TOKEN_LATENCY_S": "0.01",
            "SLM_MAX_QUEUE_SIZE": "64",
        },
        concurrency=[1, 2, 4, 8],
        requests=24,
        max_tokens=32,
        expected="throughput flat at ~1/0.36 s = 2.8 req/s; latency grows ~linearly with "
        "concurrency (C x 360 ms); ITL stays ~10 ms",
    ),
    "llama-1x": Scenario(
        description="Qwen2.5-0.5B-Instruct Q4_K_M on CPU, 1 replica x 8 threads.",
        server_env={
            "SLM_ENGINE": "llama_cpp",
            "SLM_MODEL_PATH": MODEL,
            "SLM_REPLICAS": "1",
            "SLM_N_THREADS": "8",
            "SLM_N_CTX": "2048",
        },
        concurrency=[1, 2, 4],
        requests=12,
        max_tokens=128,
        needs_model=True,
    ),
    "llama-2x": Scenario(
        description="Same model and the same 8 total threads, split into 2 replicas x 4 threads.",
        server_env={
            "SLM_ENGINE": "llama_cpp",
            "SLM_MODEL_PATH": MODEL,
            "SLM_REPLICAS": "2",
            "SLM_N_THREADS": "4",
            "SLM_N_CTX": "2048",
        },
        concurrency=[1, 2, 4],
        requests=12,
        max_tokens=128,
        needs_model=True,
    ),
    "llama-long-prompt": Scenario(
        description="Prefill cost: ~1.1k-token prompt, 16 output tokens, 1 replica x 8 threads.",
        server_env={
            "SLM_ENGINE": "llama_cpp",
            "SLM_MODEL_PATH": MODEL,
            "SLM_REPLICAS": "1",
            "SLM_N_THREADS": "8",
            "SLM_N_CTX": "2048",
        },
        concurrency=[1],
        requests=8,
        max_tokens=16,
        prompt=LONG_PROMPT,
        needs_model=True,
    ),
}


class ProcessSampler:
    """Samples a process's RSS and CPU every ``interval`` seconds, bucketed by a label."""

    def __init__(self, pid: int, interval: float = 0.25) -> None:
        self.proc = psutil.Process(pid)
        self.interval = interval
        self.label: Any = None
        self.samples: dict[Any, list[tuple[float, float]]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _tree(self) -> list[psutil.Process]:
        # Include children: on Windows a venv's python.exe is a launcher that spawns the real
        # interpreter as a child process, so the parent alone shows ~10 MB and 0% CPU.
        try:
            return [self.proc, *self.proc.children(recursive=True)]
        except psutil.Error:
            return []

    def _run(self) -> None:
        primed: dict[int, psutil.Process] = {}
        while not self._stop.wait(self.interval):
            rss_mb = cores = 0.0
            for p in self._tree():
                try:
                    if p.pid not in primed:  # first cpu_percent() call only primes the counter
                        primed[p.pid] = p
                        p.cpu_percent()
                        continue
                    rss_mb += primed[p.pid].memory_info().rss / 1024**2
                    cores += primed[p.pid].cpu_percent() / 100
                except psutil.Error:
                    continue
            if self.label is not None and rss_mb:
                self.samples.setdefault(self.label, []).append((rss_mb, cores))

    def __enter__(self) -> ProcessSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def stats(self, label: Any) -> dict[str, float | None]:
        s = self.samples.get(label, [])
        if not s:
            return {"server_peak_rss_mb": None, "server_cpu_cores_mean": None}
        return {
            "server_peak_rss_mb": round(max(r for r, _ in s), 1),
            "server_cpu_cores_mean": round(sum(c for _, c in s) / len(s), 2),
        }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run_once(name: str, sc: Scenario, run: int) -> None:
    """One trial: fresh server, warm-up, every concurrency level. Writes ``<name>/run<N>/``."""
    out_dir = RESULTS / name / f"run{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ | sc.server_env | {"SLM_PORT": str(port), "SLM_LOG_LEVEL": "WARNING"}
    print(f"[run {run}] {name}: {sc.description}")
    with open(out_dir / "server.log", "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "slm_runtime.cli", "serve"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
        try:
            load_s = _wait_ready(url, server)
            payload = loadgen.build_payload(sc.prompt, sc.max_tokens, temperature=0.0)
            levels: list[dict[str, Any]] = []
            raw: list[loadgen.RequestResult] = []
            with ProcessSampler(server.pid) as sampler:
                asyncio.run(loadgen.run_level(url, payload, 1, 2, 600))  # warmup
                for c in sc.concurrency:
                    sampler.label = c
                    results, wall = asyncio.run(
                        loadgen.run_level(url, payload, c, sc.requests, 600)
                    )
                    sampler.label = None
                    level = loadgen.summarize(results, wall) | sampler.stats(c)
                    levels.append(level)
                    raw.extend(results)
                    print(
                        f"    c={c:<3} {level['throughput_rps']} req/s  "
                        f"{level['output_tokens_per_s']} tok/s  "
                        f"p50={level['latency_ms_p50']} ms  errors={level['errors']}"
                    )
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()

    meta = {
        "scenario": name,
        "run": run,
        "description": sc.description,
        "expected": sc.expected,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "hardware": detect_hardware(),
        "platform": platform.platform(),
        "server_env": sc.server_env,
        "model_file": Path(MODEL).name if sc.needs_model else None,
        "model_size_mb": round(Path(MODEL).stat().st_size / 1024**2, 1) if sc.needs_model else None,
        "model_load_s": round(load_s, 2),
        "workload": {
            "requests_per_level": sc.requests,
            "max_tokens": sc.max_tokens,
            "temperature": 0.0,
            "prompt_chars": len(sc.prompt),
            "warmup_requests": 2,
            "client": "closed-loop, streaming",
        },
    }
    loadgen.write_results(out_dir, meta, levels, raw)


def _wait_ready(url: str, server: subprocess.Popen[bytes], timeout_s: float = 300) -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with code {server.returncode}; see server.log")
        try:
            r = httpx.get(f"{url}/readyz", timeout=2)
            if r.status_code == 200:
                return time.monotonic() - t0
            if r.json().get("status") == "failed":
                raise RuntimeError(f"model load failed: {r.json()}")
        except httpx.TransportError:
            pass
        time.sleep(0.1)
    raise TimeoutError("server did not become ready")


# Metrics reported as "median [min-max]" across runs; everything else is the median.
RANGE_METRICS = [
    "throughput_rps",
    "output_tokens_per_s",
    "ttft_ms_p50",
    "latency_ms_p50",
    "latency_ms_p95",
    "decode_tps_per_request_mean",
]
REPORT_COLUMNS = [
    "concurrency",
    "succeeded",
    "requests",
    "output_tokens_per_s",
    "throughput_rps",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "latency_ms_p50",
    "latency_ms_p95",
    "itl_ms_p50",
    "decode_tps_per_request_mean",
    "error_rate",
    "server_peak_rss_mb",
    "server_cpu_cores_mean",
]


def aggregate(name: str) -> None:
    """Combine ``<name>/run*/summary.json`` into ``<name>/summary.{json,csv}``."""
    runs = [json.loads(f.read_text()) for f in sorted((RESULTS / name).glob("run*/summary.json"))]
    if not runs:
        return
    levels = []
    for i, first in enumerate(runs[0]["levels"]):
        per_run = [r["levels"][i] for r in runs]
        level: dict[str, Any] = {"concurrency": first["concurrency"], "runs": len(runs)}
        for key, value in first.items():
            if key == "concurrency" or not isinstance(value, int | float):
                continue
            vals = [lv[key] for lv in per_run if lv.get(key) is not None]
            if not vals:
                level[key] = None
            elif key in ("requests", "succeeded"):
                level[key] = sum(vals)
            else:
                level[key] = round(statistics.median(vals), 3)
                if key in RANGE_METRICS and len(vals) > 1:
                    level[f"{key}_min"], level[f"{key}_max"] = min(vals), max(vals)
        errors: dict[str, int] = {}
        for lv in per_run:
            for k, v in lv["errors"].items():
                errors[k] = errors.get(k, 0) + v
        level["errors"] = errors
        levels.append(level)
    metas = [r["meta"] for r in runs]
    meta = metas[-1] | {
        "runs": len(runs),
        "run_timestamps": [m["timestamp"] for m in metas],
        "git_commits": sorted({m["git_commit"] for m in metas if m["git_commit"]}),
        "model_load_s": [m["model_load_s"] for m in metas],
    }
    meta.pop("run", None)
    loadgen.write_results(RESULTS / name, meta, levels, raw=None)


def _cell(level: dict[str, Any], col: str) -> str:
    v = level.get(col)
    if v is None:
        return ""
    if f"{col}_min" in level:
        return f"{v} [{level[f'{col}_min']}-{level[f'{col}_max']}]"
    return str(v)


def write_report() -> None:
    sections = [
        "# Benchmark results\n",
        "Generated by `benchmarks/run_benchmarks.py` from the `summary.json` files in this "
        "directory. Do not edit by hand. Methodology and caveats: "
        "[docs/benchmarking.md](../../docs/benchmarking.md).\n",
        "Each scenario was run several times, each with a fresh server; runs of different "
        "scenarios were interleaved. Values are the **median across runs**, with **[min-max]** "
        "for the key metrics. `succeeded`/`requests` are totals across runs. Latency, TTFT and "
        "ITL are in ms, measured client-side (HTTP, SSE and queueing included). "
        "`server_cpu_cores_mean` is the average number of CPU cores the server process kept "
        "busy. Raw per-request data is in `<scenario>/run*/requests.csv`.\n",
    ]
    for name in SCENARIOS:
        f = RESULTS / name / "summary.json"
        if not f.is_file():
            continue
        data = json.loads(f.read_text())
        m = data["meta"]
        hw = m["hardware"]
        sections.append(f"## `{name}`\n")
        sections.append(f"{m['description']}\n")
        if m.get("expected"):
            sections.append(f"*Expected if the scheduler is correct:* {m['expected']}.\n")
        env = ", ".join(f"`{k}={v}`" for k, v in m["server_env"].items() if k != "SLM_MODEL_PATH")
        wl = m["workload"]
        load = ", ".join(str(x) for x in m["model_load_s"])
        sections.append(
            f"- Runs: {m['runs']} ({m['run_timestamps'][0]} - {m['run_timestamps'][-1]}), "
            f"commit {', '.join(f'`{c}`' for c in m['git_commits'])}\n"
            f"- Host: {hw['cpu']} ({hw['cpu_cores_logical']} logical cores), {hw['ram_gb']} GB "
            f"RAM, {hw['os']}, accelerator: {hw['accelerator']}\n"
            f"- Server: {env}\n"
            + (
                f"- Model: `{m['model_file']}` ({m['model_size_mb']} MB), "
                f"load time per run: {load} s\n"
                if m.get("model_file")
                else ""
            )
            + f"- Workload per run: {wl['requests_per_level']} requests per concurrency level, "
            f"max_tokens={wl['max_tokens']}, temperature=0, prompt {wl['prompt_chars']} chars, "
            f"{wl['client']}\n"
        )
        rows = ["| " + " | ".join(REPORT_COLUMNS) + " |", "|" + "---|" * len(REPORT_COLUMNS)]
        for lv in data["levels"]:
            rows.append("| " + " | ".join(_cell(lv, c) for c in REPORT_COLUMNS) + " |")
        sections.append("\n".join(rows) + "\n")
    (RESULTS / "README.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"report written to {RESULTS / 'README.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("scenarios", nargs="*", help="Scenario names (default: all).")
    ap.add_argument("--runs", type=int, default=3, help="Trials per scenario (default: 3).")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.list:
        for n, sc in SCENARIOS.items():
            print(f"{n:<20} {'[needs model] ' if sc.needs_model else ''}{sc.description}")
        return
    if not args.report_only:
        names = args.scenarios or list(SCENARIOS)
        for name in names:
            if name not in SCENARIOS:
                sys.exit(f"unknown scenario {name!r}; use --list")
        runnable = [n for n in names if not SCENARIOS[n].needs_model or Path(MODEL).is_file()]
        for n in set(names) - set(runnable):
            print(f"[skip] {n}: model not found at {MODEL} (set SLM_BENCH_MODEL)")
        for name in runnable:  # results from a previous invocation must not mix with these
            for old in (RESULTS / name).glob("run*"):
                shutil.rmtree(old)
        # Interleave (A1 B1 A2 B2 ...) so slow periods on a shared machine hit every scenario
        # roughly equally instead of biasing whichever happened to run during them.
        for run in range(1, args.runs + 1):
            for name in runnable:
                run_once(name, SCENARIOS[name], run)
        for name in runnable:
            aggregate(name)
    write_report()


if __name__ == "__main__":
    main()
