FROM python:3.10.9


# install tesseract
# RUN apt-get update
# RUN apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu poppler-utils ffmpeg

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

ENTRYPOINT ["streamlit", "run", "agent/gui.py", "--server.port=8001", "--server.address=0.0.0.0"]
# watch the logs
# CMD ["tail", "-f", "/dev/null"]
