FROM python:3.11

# copy python installation files.
COPY ./requirements.lock  ./requirements.lock
COPY ./pyproject.toml ./pyproject.toml
COPY ./README.md ./README.md

# installing python dependencies
RUN pip install -r requirements.lock

# copy code and config files.
COPY ./config /config
COPY ./prompts /prompts
COPY ./src/agent /agent


ENTRYPOINT ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8001"]

# watch the logs
# CMD ["tail", "-f", "/dev/null"]
