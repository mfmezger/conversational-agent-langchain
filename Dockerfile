FROM python:3.11

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

# RUN pip install gpt4all-1.0.8-py3-none-manylinux1_x86_64.whl
RUN pip install gpt4all-1.0.8-py3-none-macosx_10_9_universal2.whl
# RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8001"]
# watch the logs
# CMD ["tail", "-f", "/dev/null"]
