# Makefile for Action Recognition in Videos

.PHONY: help install test run-web run-cli clean format lint type-check setup

help: ## Show this help message
	@echo "Action Recognition in Videos - Available Commands:"
	@echo "=================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

setup: ## Complete project setup
	python setup.py

test: ## Run tests
	python -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	python -m pytest tests/ --cov=src/ --cov-report=html

run-web: ## Start Streamlit web interface
	streamlit run web_app/app.py

run-example: ## Run example script
	python example.py

run-cli: ## Show CLI help
	python src/cli.py --help

format: ## Format code with black
	black src/ tests/ web_app/ example.py setup.py

lint: ## Lint code with flake8
	flake8 src/ tests/ web_app/ example.py setup.py

type-check: ## Type check with mypy
	mypy src/

clean: ## Clean temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

create-test-video: ## Create a test video
	python src/cli.py create-test --output data/test_video.mp4 --action walking

predict-video: ## Predict actions in test video (requires test video)
	python src/cli.py predict data/test_video.mp4

all-checks: format lint type-check test ## Run all code quality checks

dev-install: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Default target
.DEFAULT_GOAL := help
