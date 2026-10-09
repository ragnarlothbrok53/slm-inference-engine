"""Closed-loop load generator for an OpenAI-style /v1/completions endpoint (streaming).

Each of C concurrent workers sends a request, consumes the SSE stream, and immediately sends the
next one, until N requests have completed at that concurrency level. Timing is measured on the
client, so it includes HTTP, SSE framing and queueing - what a caller actually experiences.

Per request:   status, TTFT, end-to-end latency, completion tokens, inter-token gaps.
Per level:     success/error counts, req/s, aggregate output tok/s, p50/p95/p99 of latency and
               TTFT, p50/p95 inter-token latency (ITL), mean per-request decode tok/s.

Usage:
    python benchmarks/loadgen.py --url http://127.0.0.1:8000 --concurrency 1 2 4 \
        --requests 20 --max-tokens 128 --out benchmarks/results/manual
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_PROMPT = (
    "<|im_start|>user\nWrite a detailed, multi-paragraph essay about the history of computing, "
    "from mechanical calculators to modern GPUs.<|im_end|>\n<|im_start|>assistant\n"
)


@dataclass
class RequestResult:
    concurrency: int
    status: int | str  # HTTP status, or an error type for in-band/transport failures
    latency_s: float
    ttft_s: float | None = None
    completion_tokens: int = 0
    prompt_tokens: int = 0
    finish_reason: str | None = None
    itl_s: list[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def decode_tps(self) -> float | None:
        if self.ttft_s is None or self.completion_tokens < 2:
            return None
        decode = self.latency_s - self.ttft_s
        return (self.completion_tokens - 1) / decode if decode > 0 else None


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 100]); None for an empty list."""
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def one_request(
    client: httpx.AsyncClient, payload: dict[str, Any], concurrency: int
) -> RequestResult:
    t0 = time.perf_counter()
    res = RequestResult(concurrency=concurrency, status=0, latency_s=0.0)
    last_token_at: float | None = None
    try:
        async with client.stream("POST", "/v1/completions", json=payload) as r:
            res.status = r.status_code
            if r.status_code != 200:
                await r.aread()
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if "error" in event:
                        res.status = event["error"]["type"]
                        continue
                    now = time.perf_counter()
                    choice = event["choices"][0]
                    if choice.get("text"):
                        if res.ttft_s is None:
                            res.ttft_s = now - t0
                        elif last_token_at is not None:
                            res.itl_s.append(now - last_token_at)
                        last_token_at = now
                    if choice.get("finish_reason"):
                        res.finish_reason = choice["finish_reason"]
                    if usage := event.get("usage"):
                        res.completion_tokens = usage["completion_tokens"]
                        res.prompt_tokens = usage["prompt_tokens"]
    except httpx.HTTPError as exc:
        res.status = type(exc).__name__
    res.latency_s = time.perf_counter() - t0
    return res


async def run_level(
    url: str, payload: dict[str, Any], concurrency: int, n_requests: int, timeout_s: float
) -> tuple[list[RequestResult], float]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    results: list[RequestResult] = []
    remaining = n_requests

    async with httpx.AsyncClient(base_url=url, timeout=timeout_s, limits=limits) as client:

        async def worker() -> None:
            nonlocal remaining
            while remaining > 0:
                remaining -= 1
                results.append(await one_request(client, payload, concurrency))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    return results, wall


def summarize(results: list[RequestResult], wall_s: float) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            errors[str(r.status)] = errors.get(str(r.status), 0) + 1
    lat = [r.latency_s for r in ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    itl = [g for r in ok for g in r.itl_s]
    tps = [t for r in ok if (t := r.decode_tps) is not None]
    out_tokens = sum(r.completion_tokens for r in ok)

    def ms(v: float | None) -> float | None:
        return None if v is None else round(v * 1000, 1)

    return {
        "concurrency": results[0].concurrency if results else None,
        "requests": len(results),
        "succeeded": len(ok),
        "errors": errors,
        "error_rate": round(1 - len(ok) / len(results), 4) if results else None,
        "wall_s": round(wall_s, 3),
        "throughput_rps": round(len(ok) / wall_s, 3),
        "output_tokens_per_s": round(out_tokens / wall_s, 2),
        "mean_completion_tokens": round(out_tokens / len(ok), 1) if ok else None,
        "mean_prompt_tokens": round(sum(r.prompt_tokens for r in ok) / len(ok), 1) if ok else None,
        "latency_ms_p50": ms(percentile(lat, 50)),
        "latency_ms_p95": ms(percentile(lat, 95)),
        "latency_ms_p99": ms(percentile(lat, 99)),
        "ttft_ms_p50": ms(percentile(ttft, 50)),
        "ttft_ms_p95": ms(percentile(ttft, 95)),
        "ttft_ms_p99": ms(percentile(ttft, 99)),
        "itl_ms_p50": ms(percentile(itl, 50)),
        "itl_ms_p95": ms(percentile(itl, 95)),
        "decode_tps_per_request_mean": round(sum(tps) / len(tps), 2) if tps else None,
    }


SUMMARY_COLUMNS = [
    "concurrency",
    "succeeded",
    "requests",
    "throughput_rps",
    "output_tokens_per_s",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_p99",
    "itl_ms_p50",
    "decode_tps_per_request_mean",
    "error_rate",
]


def markdown_table(levels: list[dict[str, Any]], extra: list[str] | None = None) -> str:
    cols = SUMMARY_COLUMNS + (extra or [])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for lv in levels:
        lines.append(
            "| " + " | ".join("" if lv.get(c) is None else str(lv[c]) for c in cols) + " |"
        )
    return "\n".join(lines)


def write_results(
    out_dir: Path,
    meta: dict[str, Any],
    levels: list[dict[str, Any]],
    raw: list[RequestResult] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "levels": levels}, indent=2))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        fields = [*dict.fromkeys(k for lv in levels for k in lv if k != "errors"), "errors"]
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for lv in levels:
            w.writerow(lv | {"errors": json.dumps(lv["errors"])})
    if raw is None:
        return
    with open(out_dir / "requests.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "concurrency",
                "status",
                "latency_s",
                "ttft_s",
                "completion_tokens",
                "prompt_tokens",
                "finish_reason",
                "decode_tps",
            ]
        )
        for r in raw:
            w.writerow(
                [
                    r.concurrency,
                    r.status,
                    round(r.latency_s, 5),
                    None if r.ttft_s is None else round(r.ttft_s, 5),
                    r.completion_tokens,
                    r.prompt_tokens,
                    r.finish_reason,
                    None if r.decode_tps is None else round(r.decode_tps, 2),
                ]
            )


async def run(
    url: str,
    concurrency: list[int],
    n_requests: int,
    payload: dict[str, Any],
    warmup: int,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], list[RequestResult]]:
    if warmup:
        await run_level(url, payload, 1, warmup, timeout_s)
    levels, raw = [], []
    for c in concurrency:
        results, wall = await run_level(url, payload, c, n_requests, timeout_s)
        levels.append(summarize(results, wall))
        raw.extend(results)
        print(
            f"  concurrency={c:<3} done: {levels[-1]['throughput_rps']} req/s, "
            f"{levels[-1]['output_tokens_per_s']} tok/s, errors={levels[-1]['errors']}"
        )
    return levels, raw


def build_payload(prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": 0,
        "stream": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--requests", type=int, default=20, help="Requests per concurrency level.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", type=Path, default=None, help="Directory for JSON/CSV results.")
    args = ap.parse_args()

    payload = build_payload(args.prompt, args.max_tokens, args.temperature)
    levels, raw = asyncio.run(
        run(args.url, args.concurrency, args.requests, payload, args.warmup, args.timeout)
    )
    print(markdown_table(levels))
    if args.out:
        meta = {"url": args.url, "args": {k: str(v) for k, v in vars(args).items()}}
        write_results(args.out, meta, levels, raw)
        print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
