# Makefile
SHELL = /bin/bash

.PHONY: help style restart clean start_backend start_vectordb setup start_docker down start_frontend test

help:
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies and git hooks"
	@echo "  make style          - Run code formatting/linting"
	@echo "  make test           - Run tests with coverage"
	@echo "  make start_backend  - Start the FastAPI backend"
	@echo "  make start_frontend - Start the Streamlit frontend"
	@echo "  make start_docker   - Start all docker containers"
	@echo "  make clean          - Remove build artifacts"

setup:
	uv sync
	uv run pre-commit install

style:
	uv run pre-commit run --all-files

test:
	uv run coverage run -m pytest -v tests/
	uv run coverage report
	uv run coverage html

start_backend:
	uv run uvicorn agent.api:app --reload --port 8001

start_frontend:
	uv run streamlit run frontend/assistant.py --theme.base="dark"

start_vectordb:
	docker compose up --build -d qdrant

start_docker:
	docker compose up --build -d

restart:
	docker compose down --remove-orphans
	docker compose up --build -d

down:
	docker compose down --remove-orphans

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name "*.DS_Store" -delete
	rm -rf .coverage* htmlcov/
