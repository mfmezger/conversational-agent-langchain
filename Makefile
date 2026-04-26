# Makefile
#
# Common development shortcuts for this project. These targets wrap the
# underlying tools (`uv`, `prek`, `pytest`, `docker compose`, and Streamlit)
# so contributors can use one consistent command surface.

# Use bash for recipe execution so shell behavior is consistent across targets.
SHELL = /bin/bash

# Mark all command names as phony so make does not confuse them with files of
# the same name in the repository.
.PHONY: help style restart clean start_backend start_vectordb setup start_docker down start_frontend test test-vcr update-vcr-tests test-e2e

# Print a quick reference for the most common developer commands.
help:
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies and git hooks"
	@echo "  make style          - Run code formatting/linting"
	@echo "  make test           - Run fast tests (unit + integration) with coverage"
	@echo "  make test-vcr       - Run VCR-marked tests"
	@echo "  make update-vcr-tests - Rewrite VCR recordings"
	@echo "  make test-e2e       - Run end-to-end tests"
	@echo "  make start_backend  - Start the FastAPI backend"
	@echo "  make start_frontend - Start the Streamlit frontend"
	@echo "  make start_docker   - Start all docker containers"
	@echo "  make clean          - Remove build artifacts"

# Install backend dependencies, git hooks, and frontend dependencies.
setup:
	uv sync
	prek install
	cd frontend && uv sync

# Run the repository's configured formatting, linting, and static checks.
style:
	prek run --all-files

# Run the default test suite while excluding slower or externally-dependent
# tests. Coverage output is written both to the terminal and htmlcov/.
test:
	uv run pytest -n auto -m "not vcr and not e2e" -v --cov=src/agent --cov-report=term --cov-report=html tests/

# Run tests that rely on checked-in VCR cassettes instead of live services.
test-vcr:
	uv run pytest -m "vcr" -v tests/

# Refresh VCR cassette recordings. Use this when expected API interactions have
# changed and the checked-in recordings need to be regenerated.
update-vcr-tests:
	uv run pytest --record-mode=rewrite -m "vcr" -v tests/

# Run live end-to-end tests. These may require credentials, network access, or
# other local setup not needed by the default test target.
test-e2e:
	RUN_LIVE_E2E=1 uv run pytest -m "e2e" -v tests/

# Start the FastAPI backend in reload mode for local development.
start_backend:
	uv run uvicorn agent.api:app --reload --port 8001

# Start the Streamlit frontend from the frontend project directory.
start_frontend:
	cd frontend && uv run streamlit run assistant.py --theme.base="dark"

# Start only the vector database service needed by local backend workflows.
start_vectordb:
	docker compose up --build -d qdrant

# Start all services defined in docker-compose.yml in detached mode.
start_docker:
	docker compose up --build -d

# Rebuild and restart all Docker services from a clean compose state.
restart:
	docker compose down --remove-orphans
	docker compose up --build -d

# Stop Docker services and remove compose-managed orphan containers.
down:
	docker compose down --remove-orphans

# Remove local caches, Python bytecode, coverage output, and macOS metadata.
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name "*.DS_Store" -delete
	rm -rf .coverage* htmlcov/
