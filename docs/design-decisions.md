# Design decisions and tradeoffs

Each entry covers what was chosen, what it was weighed against, and what it costs.

## 1. Why build a serving layer instead of running `llama-server`?

llama.cpp ships `llama-server`, an HTTP server in C++ with parallel decoding slots, and for GGUF
on one machine it will out-throughput this project. vLLM, TGI and SGLang do the same on GPUs.

This project is useful where those aren't:
- It makes the **serving policy** (admission, queueing, deadlines, cancellation, readiness,
  metrics) explicit, in a few hundred lines of Python, and tested independently of any model.
  The `fake` engine lets the policy be tested and benchmarked with no model at all.
- **Backends are pluggable** behind a four-method interface (`load`, `count_tokens`, `generate`,
  `close`), so llama.cpp, MLX, or anything else that yields tokens sits behind the same API and
  the same operational behaviour.

What it costs is throughput under concurrency (see #3). At larger scale, I would keep this
project's API and policies as a thin gateway, or adopt a batching server outright (see the
README's 10x section).

## 2. Engines are synchronous; the scheduler owns all concurrency

llama.cpp and MLX expose blocking, stateful APIs. The options were:

| Option | Problem |
|---|---|
| Call the engine inside `async def` (the original code) | Blocks the event loop for the whole generation; even `/healthz` stalls. |
| `asyncio.to_thread` per request | Model objects aren't safe for concurrent use; two requests would corrupt one llama.cpp context. A lock per model fixes that but hides the queue, which you then can't bound, observe or cancel. |
| **One worker + one dedicated thread per replica, explicit queue** (chosen) | Queue depth, wait time and admission are first-class. Every call to a model happens on the same OS thread. |

Engines stay simple: a backend is about 50 lines and has no asyncio code.

## 3. No batching: one sequence per replica

`llama-cpp-python`'s high-level API decodes one sequence per context. Continuous batching would
need its low-level `llama_batch` API with several sequence IDs in one context, plus a scheduler
that interleaves prefill and decode steps. That is what `llama-server` and vLLM do.

**Consequence:** a single replica's throughput is flat as concurrency rises, and every extra
concurrent request adds a full service time to everyone's latency. The `queueing` and `llama-1x`
benchmarks show this. Batching would help on CPU too: single-sequence decode is mostly limited by
memory bandwidth (all weights are read for every token), and a batched step reads the weights
once for N sequences, until compute becomes the limit. A CPU reaches that compute limit far
sooner than a GPU, so the gain is smaller, but I haven't measured it here. On a GPU,
single-sequence decoding leaves most of the hardware idle, which is why GPU serving needs a
batching engine.

## 4. Replicas instead of batching, and what they actually buy

`SLM_REPLICAS=N` loads N independent model copies, each on its own thread. It is the simplest way
to serve more than one request at a time, but each replica holds a full copy of the weights, and
replicas compete for the same cores and memory bandwidth. Measured on a laptop CPU (README,
Performance §4): 2 replicas × 4 threads vs. 1 × 8 cut p50 TTFT at concurrency 2 from 5.3 s to
0.9 s, used 1.8× the memory, and slowed each stream (18.9 vs. 30.6 tok/s). Any aggregate
throughput gain was within run-to-run noise. Replicas buy responsiveness under light concurrency,
not capacity. Measure on the target hardware before turning them on.

## 5. Fail fast at admission (bounded queue, `503`) instead of queueing without limit

With unlimited queueing, overload turns into latency for everyone and eventually into client
timeouts, which are retried, which adds more load. A bounded queue keeps latency for accepted
work at roughly `(queue + 1) × service time` and gives callers an immediate, retryable signal.
Admission runs before the streaming response starts, so overload is a real status code even for
SSE. Setting `SLM_MAX_QUEUE_SIZE` trades rejection rate against worst-case latency, and the right
value depends on the latency SLO.

## 6. Readiness means "model loaded", not "has spare capacity"

`/readyz` returns 503 only while loading or after a failed load. Flipping readiness on saturation
looks attractive, but it pulls a busy pod from the load balancer, which pushes its traffic onto
the others, which then saturate and flip too (cascading flapping). Overload is signalled per
request (503 + `Retry-After`) and as metrics (`slm_queue_depth`, `slm_rejected_total`) for the
autoscaler.

## 7. One deadline per request, enforced on the consumer side

One `SLM_REQUEST_TIMEOUT_S` covers queue wait plus generation, because callers care about total
time. It is enforced with `asyncio.timeout_at` around the event stream rather than inside the
engine, so a stuck native call can't hold a client past the deadline. The engine is stopped
cooperatively: the replica thread checks a `threading.Event` between tokens. A single token, or a
long prefill, can't be interrupted mid-computation; that is a llama.cpp limitation.

## 8. Threads per replica, not processes

llama.cpp is called through `ctypes`, which releases the GIL during native compute, so threads
give real parallelism here, and the `overhead` benchmark shows the Python path costs under 1 ms
per token. Processes would add **fault isolation**: today a segfault or OOM in llama.cpp takes
down the whole server, including every replica. I accepted that because a container supervisor
restarts the process anyway, and one replica per container (scaling horizontally) gives the same
isolation more simply than a process pool.

## 9. Prometheus metrics; no OpenTelemetry tracing (yet)

The original dependencies listed OpenTelemetry, but nothing used it. In a single process with no
downstream calls, a trace per request would contain one span whose timings are already in the
histograms (`queue_wait`, `ttft`, `request_duration`). Tracing earns its place once there is a
request path across services (gateway → router → this server → retrieval, and so on). Then the
`X-Request-ID` propagation already in place becomes the trace context. Metrics were chosen to
answer specific questions:

- *Is it saturated?* `slm_queue_depth`, `slm_inflight_requests`, `slm_rejected_total`
- *Is the slowness queueing or the model?* `slm_queue_wait_seconds` vs
  `slm_time_to_first_token_seconds` vs `slm_decode_tokens_per_second`
- *What is it doing?* `slm_requests_total{outcome}`, token counters

## 10. OpenAI-compatible completions API, no chat endpoint yet

Matching `/v1/completions` means existing clients and load tools work as they are. Chat requires
applying the model's chat template. llama.cpp can read it from GGUF metadata, but MLX and the
fake engine would need equivalent handling, so `/v1/chat/completions` is on the roadmap rather
than half-done. `examples/simple_chat.py` applies ChatML on the client side.

## 11. Known accuracy limits

- **Token counts:** `usage.completion_tokens` counts streamed chunks. llama-cpp-python emits one
  chunk per token, except that it merges tokens that form an incomplete UTF-8 sequence, or that
  are held back as a possible stop-sequence prefix. Counts are exact for the fake engine and for
  typical English output, but can be low for non-Latin scripts.
- **Cancellation granularity:** one token (or a whole prefill) is the smallest unit that can be
  abandoned.
- **MLX backend:** written against the `mlx-lm` API but not run in CI (needs Apple Silicon).
  Stop sequences there are matched on accumulated text, so the start of a stop string spanning
  several tokens may already have been streamed.

## 12. Package layout and naming

The prototype's package was named `runtime`, with a subpackage also called `runtime`
(`runtime.runtime.engine_manager`) and 10 empty modules. It is now one flat package,
`slm_runtime`, with one module per concern. The tree is small enough that sub-packages, apart from
`engines/`, would only add indirection.
