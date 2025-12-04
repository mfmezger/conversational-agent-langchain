FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Enable bytecode compilation (faster startup)
ENV UV_COMPILE_BYTECODE=1

# Copy from cache instead of linking (required for Docker layer caching)
ENV UV_LINK_MODE=copy

# copy python installation files.
COPY ./pyproject.toml ./pyproject.toml
COPY ./README.md ./README.md
COPY ./uv.lock ./uv.lock

# installing python dependencies
RUN uv sync --frozen --no-install-project

COPY ./src /src

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.agent.api:app", "--host", "0.0.0.0", "--port", "8001"]
