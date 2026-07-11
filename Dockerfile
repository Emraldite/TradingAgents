FROM ghcr.io/astral-sh/uv:0.11.28-python3.13-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy dependency metadata first so Docker can reuse the expensive install layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev \
    && useradd --create-home appuser \
    && install -d -m 0755 -o appuser -g appuser \
        /home/appuser/.tradingagents /app/logs /app/backups \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["tradingagents"]
CMD ["--help"]
