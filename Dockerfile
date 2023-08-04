# FROM python:3.11.4
FROM ubuntu:latest

ENV PYTHON_VERSION 3.11.3
# to prevent an interactive dialog.
ENV TZ=Europe \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl git bash
#Set of all dependencies needed for pyenv to work on Ubuntu
RUN apt-get update \
        && apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget ca-certificates curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev mecab-ipadic-utf8 git

# Set-up necessary Env vars for PyEnv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install pyenv
RUN set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash
# install poetry and dependencies
# Install Poetry
RUN curl -sSL https://install.python-poetry.org/ | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy using poetry.lock* in case it doesn't exist yet
COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --no-root --no-dev

COPY . .


ENTRYPOINT ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8001"]
# watch the logs
# CMD ["tail", "-f", "/dev/null"]
