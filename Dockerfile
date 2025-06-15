FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# copy python installation files.
COPY ./pyproject.toml ./pyproject.toml
COPY ./README.md ./README.md
COPY ./uv.lock ./uv.lock

# installing python dependencies
RUN uv sync --frozen

# copy code and config files.
COPY ./config /config
COPY ./prompts /prompts
COPY ./src/agent /agent


ENTRYPOINT ["uv", "run", "uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8001"]

# watch the logs
# CMD ["tail", "-f", "/dev/null"]
