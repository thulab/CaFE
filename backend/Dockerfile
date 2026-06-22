FROM python:3.14-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.11.19@sha256:b46b03ddfcfbf8f547af7e9eaefdf8a39c8cebcba7c98858d3162bd28cf536f6 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --index-url https://mirrors.aliyun.com/pypi/simple


FROM python:3.14-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    TSBENCHMARK_RUNTIME_DIR=/var/lib/tsbenchmark \
    TSBENCHMARK_DATABASE_URL=sqlite:////var/lib/tsbenchmark/tsbenchmark.db

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY backend/app ./app
COPY backend/stub_service ./stub_service

RUN mkdir -p /var/lib/tsbenchmark

EXPOSE 8000 10810

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
