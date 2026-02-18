# Makefile
SHELL = /bin/bash

.PHONY: help style restart clean start_backend start_vectordb setup start_docker down start_frontend test test-fast test-vcr update-vcr-tests test-e2e

help:
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies and git hooks"
	@echo "  make style          - Run code formatting/linting"
	@echo "  make test           - Run fast tests (unit + integration) with coverage"
	@echo "  make test-fast      - Run fast tests (unit + integration) with coverage"
	@echo "  make test-vcr       - Run VCR-marked tests"
	@echo "  make update-vcr-tests - Rewrite VCR recordings"
	@echo "  make test-e2e       - Run end-to-end tests"
	@echo "  make start_backend  - Start the FastAPI backend"
	@echo "  make start_frontend - Start the Streamlit frontend"
	@echo "  make start_docker   - Start all docker containers"
	@echo "  make clean          - Remove build artifacts"

setup:
	uv sync
	uv run pre-commit install
	cd frontend && uv sync

style:
	uv run pre-commit run --all-files

test:
	$(MAKE) test-fast

test-fast:
	uv run pytest -n auto -m "not vcr and not e2e" -v --cov=src/agent --cov-report=term --cov-report=html tests/

test-vcr:
	uv run pytest -m "vcr" -v tests/

update-vcr-tests:
	uv run pytest --record-mode=rewrite -m "vcr" -v tests/

test-e2e:
	RUN_LIVE_E2E=1 uv run pytest -m "e2e" -v tests/

start_backend:
	uv run uvicorn agent.api:app --reload --port 8001

start_frontend:
	cd frontend && uv run streamlit run assistant.py --theme.base="dark"

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
