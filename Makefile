# Alpha Loop Capital - Makefile
# Common development commands

.PHONY: help install test lint format clean run-collect run-train run-trade

# Default target
help:
	@echo "Alpha Loop Capital - Available Commands"
	@echo "========================================"
	@echo "  install      Install dependencies"
	@echo "  test         Run test suite"
	@echo "  lint         Run linters (ruff + mypy)"
	@echo "  format       Format code with black"
	@echo "  clean        Remove cache and build artifacts"
	@echo "  run-collect  Start data collection"
	@echo "  run-train    Start model training"
	@echo "  run-trade    Start trading engine (paper mode)"

# Install dependencies
install:
	pip install -r requirements.txt

# Install dev dependencies
install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black ruff mypy

# Run tests
test:
	pytest tests/ -v --tb=short

# Run linters
lint:
	ruff check src/
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ tests/

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Run data collection
run-collect:
	python src/data_ingestion/collector.py

# Run model training
run-train:
	python src/ml/train_models.py

# Run trading engine (paper mode)
run-trade:
	python src/trading/execution_engine.py

# Windows PowerShell equivalents (use with `make -f Makefile.win`)
# For Windows, run commands directly in PowerShell

