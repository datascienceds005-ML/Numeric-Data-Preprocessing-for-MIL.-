.PHONY: help install install-dev lint format typecheck test test-fast coverage verify \
	build run run-local synthesize benchmark smoke docker-smoke determinism clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

install:         ## Install the package (runtime deps only)
	python -m pip install -e .

install-dev:     ## Install with dev/test tooling
	python -m pip install -e ".[dev]"

lint:            ## Ruff lint + format check
	ruff check src tests scripts
	ruff format --check src tests scripts

format:          ## Auto-format and fix lint
	ruff check src tests scripts --fix
	ruff format src tests scripts

typecheck:       ## mypy (strict-ish) on the package
	mypy

test-fast:       ## Fast tests, no coverage
	python -m pytest tests -q -p no:cacheprovider

coverage:        ## Full suite with the 85% gate
	python -m pytest tests -p no:cacheprovider --cov=numprep --cov-report=term-missing

test: coverage

verify:          ## All fast quality gates: lint + typecheck + full test suite
	ruff check src tests scripts
	ruff format --check src tests scripts
	mypy
	python -m pytest tests -p no:cacheprovider --cov=numprep --cov-report=term

build:           ## Build the Docker image
	docker build -t numprep:local .

synthesize:      ## Generate a seeded messy dataset into data/raw
	python -m numprep synthesize --rows 5000 --seed 42 --out data/raw

run-local:       ## Run the pipeline locally on data/raw -> outputs/run
	python -m numprep run --input data/raw --output outputs/run --seed 42

run:             ## One-command Docker reproduction (read-only input mount)
	docker run --rm \
	        -v "$$PWD/data:/data:ro" \
	        -v "$$PWD/outputs/docker-run:/out" \
	        numprep:local run --input /data --output /out

smoke: docker-smoke

docker-smoke:    ## Container smoke test incl. determinism gate (needs docker)
	bash scripts/docker_smoke_test.sh

determinism:     ## Local determinism check without docker (two runs, compare hashes)
	bash scripts/determinism_check.sh

benchmark:       ## Stage-by-stage benchmark, writes reports/benchmark.md
	python scripts/benchmark.py --sizes 10000 100000 --out reports

clean:           ## Remove caches, outputs, and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage htmlcov \
	        build dist src/*.egg-info outputs/run outputs/docker-run
	find . -name "__pycache__" -type d -exec rm -rf {} +
