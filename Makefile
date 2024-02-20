
help:
	@echo "install - install the package"
	@echo "check - run linters and formatters"
	@echo "test - run tests"
	@echo "benchmark - run benchmarks"
	@echo "coverage - run coverage"
	@echo "build - build the package"
	@echo "dist - build and upload the package"
	@echo "clean - clean the project"

clean:
	rm -rf build dist *.egg-info .pytest_cache .tox .coverage .hypothesis .mypy_cache .mypy .ruff .ruff_cache .pytest_cache .pytest .benchmarks .benchmarks_cache wheelhouse
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	
install:
	uv pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./

check:
	uv run ruff check --fix .

test: check
	uv run pytest -s -v -n auto --dist=loadfile --junitxml=tests.xml --no-cov --benchmark-disable
	
benchmark:
	uv run pytest -s -v -n 0 --no-cov benchmarks

coverage:
	uv run pytest --cov=sources --cov-report=html --cov-report=xml --benchmark-disable

build: test
	python -m build --wheel

compile:
	python -m pyc_wheel dist/iopathlib-*.whl --optimize 2

dist: 
	uv run twine check dist/*
	uv run twine upload dist/*

.PHONY: help install check test benchmark coverage dist build clean compile