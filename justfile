# linkml-store justfile

# List all commands as default
_default:
    @just --list

# ============== Installation ==============

# Install project dependencies (dev + all extras)
install:
    uv sync --group dev --all-extras

# Install minimal dependencies
install-minimal:
    uv sync

# ============== Testing ==============

# Run all tests (excluding integration and neo4j)
test:
    uv run pytest -m "not integration and not neo4j" tests

# Run core tests
test-core: pytest doctest

# Run full test suite
test-full: pytest-full doctest

# Run pytest (excluding integration tests)
pytest:
    uv run pytest -m "not integration" tests

# Run neo4j tests
pytest-neo4j:
    uv run pytest -k only_neo4j

# Run minimal pytest
pytest-minimal:
    uv run pytest tests/test_api/test_filesystem_adapter.py

# Run full pytest (all markers)
pytest-full:
    uv run pytest -m ""

# Run integration tests
integration-tests:
    uv run pytest -m integration

# Run all pytest tests
all-pytest:
    uv run pytest -m "integration or not integration"

# Run doctests
doctest:
    find docs src/linkml_store/api src/linkml_store/index src/linkml_store/utils -type f \( -name "*.rst" -o -name "*.md" -o -name "*.py" \) ! -path "*/chromadb/*" -print0 | xargs -0 uv run python -m doctest --option ELLIPSIS --option NORMALIZE_WHITESPACE

# ============== Linting ==============

# Run ruff check
lint:
    uv run ruff check .

# Run ruff format check
format-check:
    uv run ruff format --check .

# Format code with ruff
format:
    uv run ruff format .

# Run mypy
mypy:
    uv run mypy src tests

# ============== Apps ==============

# Run the API server
api:
    uv run linkml-store-api

# Run the streamlit app
app:
    uv run streamlit run src/linkml_data_browser/app.py

# ============== Documentation ==============

# Build sphinx docs
docs-html:
    cd docs && uv run make html

# Clean and build docs
docs-clean:
    cd docs && uv run make clean html

# ============== Utilities ==============

# Run codespell
codespell:
    uv run codespell

# Lock dependencies
lock:
    uv lock

# Update dependencies
update:
    uv lock --upgrade
