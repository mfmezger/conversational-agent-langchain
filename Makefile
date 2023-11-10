# Makefile
SHELL = /bin/bash

.PHONY : style
style:
	pre-commit run --all-files

restart:
	docker compose down --remove-orphans
	docker compose up --build -d

.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf .coverage*

start_backend:
	poetry run uvicorn app.main:app --reload --port 8001

start_vectordb:
	docker compose up --build -d qdrant

setup:
	poetry install

start_docker:
	docker compose up --build -d

down:
	docker compose down --remove-orphans

start_frontend:
	poetry run streamlit run asdf.py --theme.base="dark"

test:
	poetry run coverage run -m pytest -o log_cli=true -vvv tests/
	poetry run coverage report
	poetry run coverage html
