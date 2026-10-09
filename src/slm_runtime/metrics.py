"""Prometheus metrics. Each app instance owns its registry so tests stay isolated.

The metric set is chosen to answer the questions an on-call engineer actually asks of an LLM
server: *Is it saturated?* (queue_depth, inflight, rejected), *is it slow because of queueing or
because of the model?* (queue_wait vs ttft vs decode tokens/s), and *what is it producing?*
(token counters, request outcomes).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_TPS_BUCKETS = (1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "slm_requests_total",
            "Completion requests by outcome (ok|timeout|cancelled|engine_error|shutdown).",
            ["outcome"],
            registry=r,
        )
        self.rejected = Counter(
            "slm_rejected_total", "Requests rejected at admission.", ["reason"], registry=r
        )
        self.request_duration = Histogram(
            "slm_request_duration_seconds",
            "Submit-to-last-token latency of successful requests (includes queue wait).",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.queue_wait = Histogram(
            "slm_queue_wait_seconds",
            "Time spent waiting for a free replica.",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "slm_time_to_first_token_seconds",
            "Submit-to-first-token latency (queue wait + prefill).",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.decode_tokens_per_second = Histogram(
            "slm_decode_tokens_per_second",
            "Per-request decode throughput, excluding time to first token.",
            buckets=_TPS_BUCKETS,
            registry=r,
        )
        self.generated_tokens = Counter(
            "slm_generated_tokens_total", "Completion tokens produced.", registry=r
        )
        self.prompt_tokens = Counter(
            "slm_prompt_tokens_total", "Prompt tokens processed.", registry=r
        )
        self.queue_depth = Gauge("slm_queue_depth", "Requests waiting for a replica.", registry=r)
        self.inflight = Gauge("slm_inflight_requests", "Requests being generated.", registry=r)
        self.replicas = Gauge("slm_replicas", "Loaded engine replicas.", registry=r)
