# Build stage
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Create data and logs directories
RUN mkdir /data /logs

# Copy python installation files
COPY ./pyproject.toml ./uv.lock ./README.md ./
# Installing python dependencies
RUN uv sync --frozen --no-install-project --no-dev

COPY ./src ./src
RUN uv sync --frozen --no-dev

# Final stage
FROM cgr.dev/chainguard/python:latest@sha256:6b75c6363142679d082d3aed51280bd7eace8412e9ce7ab8394e38fc80ae9889

WORKDIR /app

# Copy the virtual environment and source code from the builder
COPY --from=builder --chown=65532:65532 /app/.venv /app/.venv
COPY --from=builder --chown=65532:65532 /app/src /app/src
COPY --from=builder --chown=65532:65532 /data /data
COPY --from=builder --chown=65532:65532 /logs /logs

# Use the virtual environment's python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "uvicorn", "src.agent.api:app", "--host", "0.0.0.0", "--port", "8001"]
