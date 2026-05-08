# syntax=docker/dockerfile:1.7

# ----------------------------------------------------------------------------
# Stage 1: build the wheel
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN pip install --upgrade pip build

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

RUN python -m build --wheel --outdir /wheels

# ----------------------------------------------------------------------------
# Stage 2: runtime
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/hf_cache/sentence-transformers

WORKDIR /app

# CPU-only torch keeps the image ~1 GB lighter than the default CUDA wheel.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0"

COPY --from=builder /wheels/*.whl /tmp/
RUN whl="$(ls /tmp/*.whl)" && pip install "${whl}[web]" && rm /tmp/*.whl

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/embeddings_cache /app/web_settings /app/hf_cache \
    && chown -R app:app /app

USER app

VOLUME ["/app/embeddings_cache", "/app/web_settings", "/app/hf_cache"]

EXPOSE 8000

ENTRYPOINT ["esco-skill-extractor"]
CMD ["web", "--host", "0.0.0.0", "--port", "8000"]
