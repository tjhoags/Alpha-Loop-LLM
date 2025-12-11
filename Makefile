# ================================================================================
# Alpha Loop Capital - Makefile
# ================================================================================
# Author: Tom Hogan | Alpha Loop Capital, LLC
# 
# Cross-platform development and operations commands.
# Usage: make <target>
# ================================================================================

.PHONY: help install dev-install clean lint format test collect train trade
.DEFAULT_GOAL := help

# ================================================================================
# Variables
# ================================================================================
PYTHON := python
PIP := pip
VENV := venv
SRC := src
SCRIPTS := scripts

# Windows detection
ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE := $(VENV)\Scripts\activate
    RM := del /Q /F
    RMDIR := rmdir /S /Q
    SEP := \\
else
    VENV_ACTIVATE := source $(VENV)/bin/activate
    RM := rm -f
    RMDIR := rm -rf
    SEP := /
endif

# ================================================================================
# Help
# ================================================================================
help:  ## Show this help message
	@echo.
	@echo Alpha Loop Capital - Development Commands
	@echo ==========================================
	@echo.
	@echo Setup:
	@echo   make install       - Install production dependencies
	@echo   make dev-install   - Install development dependencies
	@echo   make venv          - Create virtual environment
	@echo.
	@echo Operations:
	@echo   make collect       - Run data collection cycle
	@echo   make train         - Run model training
	@echo   make trade         - Start trading engine (simulation)
	@echo   make trade-live    - Start trading engine (LIVE - CAUTION)
	@echo.
	@echo Development:
	@echo   make lint          - Run linters
	@echo   make format        - Format code
	@echo   make test          - Run tests
	@echo   make clean         - Clean build artifacts
	@echo.

# ================================================================================
# Setup
# ================================================================================
venv:  ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	@echo Virtual environment created. Activate with:
	@echo   Windows: .$(SEP)venv$(SEP)Scripts$(SEP)Activate.ps1
	@echo   Mac/Linux: source venv/bin/activate

install:  ## Install production dependencies
	$(PIP) install -r requirements.txt

dev-install:  ## Install development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

upgrade-deps:  ## Upgrade all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt

# ================================================================================
# Operations
# ================================================================================
collect:  ## Run data collection
	$(PYTHON) $(SRC)/data_ingestion/collector.py

train:  ## Run model training
	$(PYTHON) $(SRC)/ml/train_models.py

train-massive:  ## Run massive trainer (full universe)
	$(PYTHON) $(SRC)/ml/massive_trainer.py

train-agents:  ## Train all agents
	$(PYTHON) $(SRC)/training/agent_trainer.py

trade:  ## Start trading engine (simulation mode)
	$(PYTHON) $(SRC)/trading/execution_engine.py --simulation

trade-live:  ## Start trading engine (LIVE - CAUTION)
	@echo ============================================
	@echo WARNING: LIVE TRADING MODE
	@echo This will execute REAL trades with REAL money
	@echo ============================================
	@echo Press Ctrl+C to cancel...
	@timeout 5
	$(PYTHON) $(SRC)/trading/execution_engine.py

dashboard:  ## Start model dashboard
	$(PYTHON) $(SCRIPTS)/model_dashboard.py

hydrate:  ## Hydrate full universe data
	$(PYTHON) $(SCRIPTS)/hydrate_full_universe.py

# ================================================================================
# Development
# ================================================================================
lint:  ## Run linters (ruff + mypy)
	ruff check $(SRC) $(SCRIPTS)
	mypy $(SRC) --ignore-missing-imports

format:  ## Format code with black and ruff
	black $(SRC) $(SCRIPTS)
	ruff check --fix $(SRC) $(SCRIPTS)

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=$(SRC) --cov-report=html --cov-report=term

test-fast:  ## Run fast tests only
	pytest tests/ -v -m "not slow and not integration"

# ================================================================================
# Database
# ================================================================================
db-setup:  ## Setup database schema
	$(PYTHON) $(SCRIPTS)/setup_db_schema.py

db-test:  ## Test database connection
	$(PYTHON) $(SCRIPTS)/test_db_connection.py

# ================================================================================
# API Testing
# ================================================================================
test-apis:  ## Test all API connections
	$(PYTHON) $(SCRIPTS)/test_all_apis.py

test-api-connections:  ## Quick API connection test
	$(PYTHON) $(SCRIPTS)/test_api_connections.py

# ================================================================================
# Cleanup
# ================================================================================
clean:  ## Clean build artifacts
	$(RMDIR) __pycache__ 2>nul || true
	$(RMDIR) .pytest_cache 2>nul || true
	$(RMDIR) .ruff_cache 2>nul || true
	$(RMDIR) .mypy_cache 2>nul || true
	$(RMDIR) htmlcov 2>nul || true
	$(RMDIR) .coverage 2>nul || true
	$(RMDIR) build 2>nul || true
	$(RMDIR) dist 2>nul || true
	$(RMDIR) *.egg-info 2>nul || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-models:  ## Clean trained models (CAUTION)
	@echo WARNING: This will delete all trained models!
	$(RMDIR) models$(SEP)*.pkl 2>nul || true

clean-logs:  ## Clean log files
	$(RMDIR) logs$(SEP)*.log 2>nul || true

clean-all: clean clean-logs  ## Clean everything except models

# ================================================================================
# Git Operations
# ================================================================================
status:  ## Show git status
	git status

commit:  ## Commit changes (interactive)
	git add -A
	git commit

push:  ## Push to origin
	git push origin $(shell git branch --show-current)

pull:  ## Pull from origin
	git pull origin $(shell git branch --show-current)

# ================================================================================
# Deployment
# ================================================================================
build:  ## Build package
	$(PYTHON) -m build

publish-test:  ## Publish to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

# ================================================================================
# Overnight Operations (Windows)
# ================================================================================
overnight-windows:  ## Start overnight training (Windows)
	powershell -ExecutionPolicy Bypass -File $(SCRIPTS)\overnight_training_robust.ps1

fullthrottle-windows:  ## Start full throttle training (Windows)
	powershell -ExecutionPolicy Bypass -File $(SCRIPTS)\start_full_throttle_training.ps1

