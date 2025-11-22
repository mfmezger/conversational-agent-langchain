# Makefile
SHELL = /bin/bash

style:
	pre-commit run --all-files

restart:
	docker compose down --remove-orphans
	docker compose up --build -d

clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf .coverage*

start_backend:
	uv run uvicorn agent.api:app --reload --port 8001

start_vectordb:
	docker compose up --build -d qdrant

setup:
	uv sync

start_docker:
	docker compose up --build -d

down:
	docker compose down --remove-orphans

start_frontend:
	uv run streamlit run frontend/assistant.py --theme.base="dark"

test:
	uv run coverage run -m pytest -o log_cli=true -vvv tests/
	uv run coverage report
	uv run coverage html
