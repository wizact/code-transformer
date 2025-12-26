.PHONY: help test test-unit test-integration test-cov lint format type-check clean install dev

help:
	@echo "Code Transformer - Development Commands"
	@echo ""
	@echo "Testing:"
	@echo "  make test              Run all tests"
	@echo "  make test-unit         Run unit tests only"
	@echo "  make test-integration  Run integration tests only"
	@echo "  make test-cov          Run tests with coverage report"
	@echo "  make test-cov-html     Run tests with HTML coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              Run ruff linter"
	@echo "  make format            Format code with ruff"
	@echo "  make type-check        Run mypy type checker"
	@echo "  make quality           Run all quality checks (lint + type-check)"
	@echo ""
	@echo "Development:"
	@echo "  make install           Install production dependencies"
	@echo "  make dev               Install development dependencies"
	@echo "  make clean             Remove build artifacts and cache"
	@echo ""

# Testing
test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-cov:
	uv run pytest tests/ --cov=code_transformer --cov-report=term-missing

test-cov-html:
	uv run pytest tests/ --cov=code_transformer --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Code Quality
lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

type-check:
	uv run mypy src/

quality: lint type-check
	@echo "✓ All quality checks passed"

# Development
install:
	uv sync --no-dev

dev:
	uv sync --dev

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
