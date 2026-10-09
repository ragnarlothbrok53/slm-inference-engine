# Benchmarking methodology

Results live in [`benchmarks/results/`](../benchmarks/results/README.md). This page explains how
they are produced and how to read them, including what they do **not** show.

## What is measured, and where

All timings are taken **client-side** by [`benchmarks/loadgen.py`](../benchmarks/loadgen.py), so
they include HTTP, SSE framing, the scheduler queue and the model. That is what a caller
experiences. Server-side histograms with the same breakdown are exposed on `/metrics`.

| Metric | Definition |
|---|---|
| **TTFT** | Request sent → first non-empty token received. Includes queue wait and prompt prefill. |
| **Latency** | Request sent → `[DONE]` received (end-to-end). |
| **ITL** | Gap between consecutive token events on one stream (inter-token latency). |
| **decode tok/s (per request)** | `(completion_tokens − 1) / (latency − TTFT)`: the streaming speed a single user sees. |
| **output tok/s (aggregate)** | Total completion tokens across successful requests / wall time of the level: server throughput. |
| **req/s** | Successful requests / wall time. |
| **error_rate** | Non-200 responses, in-band stream errors or transport errors / requests. |
| **server_peak_rss_mb** | Peak resident memory of the server process tree, sampled every 250 ms. |
| **server_cpu_cores_mean** | Mean CPU use of the server process tree, in cores (1.0 = one core fully busy). |

Percentiles use linear interpolation. With 12–24 requests per level, **p99 is effectively the max**
and p95 is close to it. Read the tails as indicative, not statistically tight.

## Workload

- **Closed loop.** `C` workers each send a request, read the whole stream, then send the next
  one, until `N` requests finish at that concurrency. Offered load therefore adapts to the
  server's speed. That suits measuring capacity and the latency/concurrency tradeoff, but it does
  **not** model open-loop (Poisson) arrival traffic, so it can't show behaviour beyond saturation.
- **Deterministic generation.** `temperature=0`, `seed=0`, and no stop sequences. The model may
  still emit EOS early, so token counts come from each response's `usage` field, not from
  `max_tokens`.
- **Unique prompt prefix per request** (default). llama.cpp reuses the KV cache for the longest
  prefix a prompt shares with the previous one on the same context. If the harness repeats one
  prompt verbatim, every request after the first is a cache hit, and TTFT measures the cache, not
  prefill. The first version of this harness had exactly that bug: it reported 76 ms TTFT for a
  prompt that takes ~10 s to prefill cold. Each request now starts with a random tag;
  `llama-long-prompt-cached` (or `loadgen.py --repeat-prompt`) turns that off on purpose.
- **Warm-up.** Two requests run before measurement begins (page-in of weights, allocator warm-up).
- **Repeated, interleaved trials.** `--runs N` (default 3) repeats every scenario with a fresh
  server each time, in the order A1 B1 … A2 B2 …, so a slow period on a shared machine affects
  every scenario rather than biasing one. Reports show the median across runs and [min–max] for
  key metrics; every run's raw data is kept in `<scenario>/runN/`.

## Scenarios

| Scenario | Purpose | Needs model |
|---|---|---|
| `overhead` | Fake engine with zero latency: isolates the cost of the serving stack itself (HTTP, SSE, scheduler, thread hand-off). | no |
| `queueing` | Fake engine with a known 360 ms service time on 1 replica: checks the scheduler against queueing theory (flat throughput, latency ∝ concurrency). | no |
| `llama-1x` | Real model, 1 replica × 8 threads, concurrency 1/2/4. | yes |
| `llama-2x` | Same model and the same 8 threads total, split as 2 replicas × 4 threads. | yes |
| `llama-long-prompt` | ~1.2k-token prompt (unique per request), 16 output tokens: cold prefill cost shows up in TTFT. | yes |
| `llama-long-prompt-cached` | Same prompt sent verbatim every time: shows what KV prefix reuse saves. | yes |

The default model is **Qwen2.5-0.5B-Instruct, Q4_K_M GGUF** (~470 MB) from the official
[`Qwen/Qwen2.5-0.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) repo.
Override with `SLM_BENCH_MODEL=/path/to/model.gguf`; the prompts use the ChatML template.

## Reproducing

```bash
uv sync --extra llama
mkdir -p models
curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

uv run python benchmarks/run_benchmarks.py --list
uv run python benchmarks/run_benchmarks.py                  # all scenarios (~10 min on a laptop CPU)
uv run python benchmarks/run_benchmarks.py overhead queueing # no model needed
```

Each scenario writes `summary.json` (metadata + per-level aggregates), `summary.csv` and
`requests.csv` (one row per request), then regenerates `benchmarks/results/README.md`.
`summary.json` records the git commit (with `-dirty` if `src/` had uncommitted changes), host
hardware, server configuration and workload, so a result can be traced back to how it was made.

To benchmark a server you started yourself (any config, or another OpenAI-compatible server):

```bash
uv run python benchmarks/loadgen.py --url http://127.0.0.1:8000 \
  --concurrency 1 2 4 8 --requests 20 --max-tokens 128 --out benchmarks/results/manual
```

## Caveats that apply to the committed results

- **Laptop, not a lab.** The committed numbers come from a Windows laptop (Intel Core Ultra 7
  165U: 2 performance + 8 efficient + 2 low-power cores, 14 threads) running normal background
  software, with thermal and power management left on. Measured spread: under 1% run to run for
  `queueing` (the fake engine sleeps, so the host barely matters), but up to about 2.5× between
  runs for real-model aggregate throughput (e.g. 17.6–46.7 tok/s for `llama-2x` at concurrency
  2). Per-request decode speed also drifted within single runs as the CPU's power state changed.
  Read the real-model numbers as ranges, compare only scenarios that ran interleaved on the same
  machine, and don't compare across machines.
- **Client and server share the CPU.** The load generator is also Python on the same host. In
  `overhead` this matters a lot: at high concurrency the client competes for cores, so treat those
  numbers as a lower bound on what the server can do.
- **ITL in `overhead` is meaningless.** With zero model latency, tokens are produced faster than
  they're delivered, so many SSE events arrive in one TCP read and gaps read as ~0 ms.
- **CPU only.** No GPU numbers are reported because none were measured.
