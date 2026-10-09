# syntax=docker/dockerfile:1
# CPU image with the llama.cpp backend. The model is NOT baked in: mount it at /models.
#
#   docker build -t slm-runtime .
#   docker run --rm -p 8000:8000 -v "$PWD/models:/models:ro" \
#     -e SLM_MODEL_PATH=/models/qwen2.5-0.5b-instruct-q4_k_m.gguf slm-runtime

FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so source edits don't invalidate the (slow) dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra llama --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra llama --no-editable

FROM python:3.11-slim
# libgomp: OpenMP runtime used by the prebuilt llama.cpp wheel.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=builder /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SLM_HOST=0.0.0.0 \
    SLM_PORT=8000 \
    SLM_LOG_JSON=true \
    SLM_MODEL_PATH=/models/model.gguf
USER app
WORKDIR /home/app
EXPOSE 8000
# "healthy" means ready to serve (weights loaded); allow time for large models to load.
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)"]
CMD ["slm-runtime", "serve"]
