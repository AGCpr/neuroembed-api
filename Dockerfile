# syntax=docker/dockerfile:1.7
# Multi-stage Dockerfile. The final image is small (no torch) by default.
# To build the GPU variant: `docker build --target gpu .`
ARG PYTHON_VERSION=3.11

# ---- base ----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# ---- cpu (default) -------------------------------------------------------
FROM base AS cpu
# Copy metadata files that pyproject.toml references (readme, license).
# The src/ copy happens after pip install so changes to source don't
# bust the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[dev]" && \
    useradd --create-home --shell /bin/bash app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["neuroembed-api", "--host", "0.0.0.0", "--port", "8000"]

# ---- gpu -----------------------------------------------------------------
# Use a CUDA base for the worker process. torch + transformers + mne are
# installed in this target only, keeping the cpu image small.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

ARG PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python3-pip curl \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[dev,model]"
USER root
CMD ["neuroembed-api", "--host", "0.0.0.0", "--port", "8000"]
