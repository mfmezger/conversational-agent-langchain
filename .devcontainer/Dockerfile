# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.0/containers/python-3/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/devcontainers/python:3.12

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for all users
ENV PATH="/root/.local/bin:$PATH"
