UV_RUN = uv run
PYTHON = python -m
SRC = src
EXCLUDE = .venv,__pycache__,.mypy_cache,llm_sdk

install:
	uv sync

run:
	$(UV_RUN) $(PYTHON) $(SRC)

debug:
	$(UV_RUN) $(PYTHON) pdb -m $(SRC)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -name "*.pyc" -delete

lint:
	$(UV_RUN) flake8 . --exclude=$(EXCLUDE)
	$(UV_RUN) mypy src/ --no-namespace-packages --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV_RUN) flake8 . --exclude=$(EXCLUDE)
	$(UV_RUN) mypy src/ --strict
