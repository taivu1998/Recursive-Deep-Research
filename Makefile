.PHONY: setup install clean format test inference evaluate train venv

# Python executable (use venv if available)
PYTHON := $(shell if [ -d "venv" ]; then echo "venv/bin/python"; else echo "python3.10"; fi)
PIP := $(shell if [ -d "venv" ]; then echo "venv/bin/pip"; else echo "pip"; fi)

# Create virtual environment
venv:
	python3.10 -m venv venv
	$(PIP) install --upgrade pip

# Full setup: create venv and install dependencies
setup: venv install
	@echo "Setup complete! Don't forget to:"
	@echo "  1. Edit .env with your API keys"
	@echo "  2. Run 'make inference' to test"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf logs/*
	rm -rf checkpoints/*

format:
	$(PYTHON) -m black src scripts
	$(PYTHON) -m isort src scripts

test:
	$(PYTHON) -m pytest tests/

# Run single inference query
inference:
	$(PYTHON) scripts/inference.py --config configs/default.yaml --query "Compare the revenue growth of NVIDIA vs AMD in the quarter after H100 release."

# Run single inference with custom query
inference-custom:
	@read -p "Enter your query: " query; \
	$(PYTHON) scripts/inference.py --config configs/default.yaml --query "$$query"

# Run full evaluation on golden set
evaluate:
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml

# In Agentic workflows, 'training' is often synonymous with 'running the evaluation harness'
train: evaluate

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup      - Create venv and install all dependencies"
	@echo "  make install    - Install dependencies only"
	@echo "  make inference  - Run single query inference"
	@echo "  make evaluate   - Run full evaluation on golden set"
	@echo "  make clean      - Clean cache and logs"
	@echo "  make format     - Format code with black and isort"