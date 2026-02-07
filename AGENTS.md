# Agent Development Guide

This file contains guidelines for agentic coding agents working in this repository.

## Development Commands

### Setup
```bash
make setup              # Install dependencies and git hooks
uv sync                 # Install/refresh dependencies
```

### Linting and Formatting
```bash
make style              # Run code formatting/linting (pre-commit hooks)
uv run pre-commit run --all-files  # Run all pre-commit hooks manually
uv run ruff check src/  # Run linter
uv run ruff format src/ # Run formatter
```

### Testing
```bash
make test               # Run all tests with coverage
uv run coverage run -m pytest -v tests/
uv run coverage report  # View coverage report
```

### Running a Single Test
```bash
# Run specific test file
uv run pytest tests/unit_tests/test_utility.py -v

# Run specific test function
uv run pytest tests/unit_tests/test_utility.py::test_combine_text_from_list -v

# Run tests in a directory
uv run pytest tests/unit_tests/ -v
```

### Running the Application
```bash
make start_backend      # Start FastAPI backend on port 8001
make start_frontend     # Start Streamlit frontend on port 8501
make start_docker       # Start all docker containers (Qdrant, Phoenix)
```

## Code Style Guidelines

### Imports
- Group imports: standard library → third-party → local
- Use `from collections.abc import` for type annotations (AsyncGenerator, Sequence, etc.)
- Use `from typing import` for typing utilities only (Annotated, Literal, TypedDict)
- Separate import groups with blank lines
- Sort imports alphabetically within groups

### Formatting
- **Line length**: 170 characters
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings and docstrings
- **Trailing commas**: Use for multi-line structures

### Type Hints
- Python 3.13+ target (3.11 minimum)
- Use modern type syntax: `list[T]` instead of `List[T]`, `dict[K, V]` instead of `Dict[K, V]`
- Optional types: `T | None` instead of `Optional[T]`
- Use `Annotated` for metadata: `Annotated[list[BaseMessage], add_messages]`
- Use `Literal` for string literal unions
- All functions must have type hints (mypy enforces: `disallow_untyped_defs`)

### Naming Conventions
- **Classes**: PascalCase (`AgentState`, `Graph`, `Config`)
- **Functions/Methods**: snake_case (`retrieve_documents`, `build_graph`)
- **Variables**: snake_case (`query`, `documents`, `retry_count`)
- **Constants**: UPPER_SNAKE_CASE (`GEMINI_MODEL_KEY`, `COHERE_MODEL_KEY`)
- **Private members**: underscore prefix (`_write_file_to_disk`)
- **Modules**: snake_case lowercase (`internal_model.py`)

### Error Handling
- Use `try/except` blocks with specific exception types
- API routes: raise `fastapi.HTTPException` with status codes and details
- Log errors using `loguru.logger`: `logger.error(f"Error message: {exc}")`
- Validate input data using Pydantic models

### Logging
- Use `loguru` logger: `from loguru import logger`
- Log levels: `logger.info()`, `logger.warning()`, `logger.error()`, `logger.debug()`
- Include context in log messages

### Docstrings
- Use triple double quotes for docstrings
- One-line summary for simple functions/classes
- Multi-line docstrings for complex components
- Describe purpose, parameters, and return values

### Async/Await
- Use `async def` for async functions
- Use `await` for async calls
- Use `AsyncMock` from `unittest.mock` for testing async code
- Use `unittest.IsolatedAsyncioTestCase` for async test classes

### Testing
- Use `pytest` as test framework
- Use `unittest.IsolatedAsyncioTestCase` for async tests
- Use `unittest.mock.MagicMock` and `AsyncMock` for mocking
- Use `pytest.raises()` for expected exceptions
- Test naming: `test_<function_name>_<scenario>`

### Configuration
- Environment variables: use `.env` file
- Configuration loading: via `agent.utils.config.Config` (Pydantic BaseSettings)
- Never commit `.env` file (it's in `.gitignore`)

### Project Structure
```
src/agent/
├── __init__.py
├── api.py                    # FastAPI app definition
├── backend/
│   ├── graph.py              # LangGraph state machine
│   └── services/
├── data_model/
│   ├── internal_model.py     # Internal Pydantic models
│   ├── request_data_model.py # Request models
│   └── response_data_model.py# Response models
├── routes/                   # FastAPI route handlers
├── utils/
│   ├── config.py             # Configuration via Pydantic
│   ├── retriever.py          # Document retrieval
│   ├── reranker.py           # Reranking logic
│   └── vdb.py                # Vector database operations
└── scripts/                  # Utility scripts

tests/
├── unit_tests/
├── e2e_tests/
├── test_integration.py
└── test_stream.py
```

### Additional Notes
- This project uses LangGraph for agent orchestration
- Vector database: Qdrant (runs in Docker)
- Observability: Phoenix tracing
- Reranking: Cohere (cloud) or FlashRank (local)
- Before committing changes, run `make style` to ensure code passes linting/formatting
- After committing, mypy will check type hints automatically
