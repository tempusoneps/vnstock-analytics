# syntax=docker/dockerfile:1.7

FROM python:3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CONCURRENT_DOWNLOADS=4 \
    UV_HTTP_RETRIES=10 \
    UV_HTTP_TIMEOUT=120 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /notebooks

COPY ./src/requirements.txt .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

ARG INSTALL_LABEL_OHLCV=true

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install git+https://github.com/tempusoneps/autofcholv.git@develop

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install git+https://github.com/tempusoneps/labelohlcv.git@develop
