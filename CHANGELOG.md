# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [0.1.0] - 2026-10-09

First tagged release. It replaces the initial prototype (`src/runtime/`), which had a single
blocking completion route and placeholder modules.

### Added
- `Scheduler`: bounded FIFO admission (`503` + `Retry-After` when full), one dedicated thread per
  model replica, per-request deadlines, cancellation on timeout/disconnect, graceful shutdown.
- OpenAI-compatible `POST /v1/completions` with SSE streaming (`stream: true`), `stop`, `seed`,
  `temperature`, `top_p`, `usage`; `GET /v1/models`.
- `GET /healthz` (liveness) and `GET /readyz` (readiness; the model loads in the background).
- Prometheus `GET /metrics`: TTFT, queue wait, request duration, decode tokens/s, token counters,
  queue depth, in-flight requests, rejections, outcomes.
- Structured logging (text or JSON) with `X-Request-ID` propagation and one access-log line per
  request; per-generation log line with TTFT, queue wait and token counts.
- `fake` engine with configurable latency, for tests and serving-overhead benchmarks.
- `SLM_REPLICAS`: N independent model instances in one process.
- Configuration through `SLM_*` environment variables / `.env` (pydantic-settings).
- Benchmark harness (`benchmarks/`): closed-loop streaming load generator, multi-run interleaved
  scenarios, JSON/CSV output, generated results report.
- Test suite: scheduler unit tests, HTTP contract tests, real-socket disconnect test, opt-in
  real-model test.
- GitHub Actions CI (lint, format, mypy strict, tests, real-model integration test, Docker build
  and smoke test), Dockerfile, `.env.example`, MIT `LICENSE`.

### Fixed
- Inference ran synchronously inside `async` handlers, blocking the event loop (and every other
  request, including health checks) for the whole generation.
- The request's `max_tokens` was ignored (always 128).
- `mlx-lm` was imported unconditionally and listed as a hard dependency, which broke install and
  startup on Linux and Windows. Backends are now optional extras, imported lazily.
- An unknown `ENGINE` value left the engine unset, failing on the first request. It is now
  rejected at startup by config validation.

### Changed
- Package renamed `runtime` → `slm_runtime`; distribution `ml-service` → `slm-runtime`.
- Hardware detection no longer imports PyTorch; `torch`, `transformers` and the unused
  OpenTelemetry packages are no longer dependencies.
- CLI: `slm-runtime serve` (flags override env vars) and `slm-runtime info`.

### Removed
- Empty placeholder modules (`router.py`, `profiler.py`, `token_streamer.py`, `settings.py`, …)
  and the unused `main.py`.
