all: format lint xtest

lint: FORCE  ## Run isort, flake8, mypy and black (in check mode)
	poetry run isort -rc --check-only .
	poetry run flake8
	poetry run mypy .
	poetry run black . --check

test: FORCE  ## Run pytest
	poetry run python -m pytest --cov=ergo --ff --verbose -s --doctest-modules .

test_skip_metaculus: FORCE  ## Run pytest, but skip the Metaculus tests to avoid overburdening the Metaculus API
	poetry run python -m pytest --cov=ergo --ff --verbose -s --doctest-modules --ignore-glob='*test_metaculus.py' .

xtest: FORCE  ## Run pytest in parallel mode using xdist
	poetry run python -m pytest --cov=ergo --ff --verbose -s --doctest-modules -n auto .

format: FORCE  ## Run isort and black (rewriting files)
	poetry run isort -rc .
	poetry run black .

docs: FORCE  ## Build docs
	poetry run $(MAKE) -C docs html

serve: FORCE  ## Run Jupyter notebook server
	poetry run python -m jupyter lab

scrub: FORCE  ## Create scrubbed notebooks in notebooks/scrubbed from notebooks
	poetry run python scripts/scrub_notebooks.py notebooks notebooks/scrubbed

scrub_src_only: FORCE  ## Scrub notebooks in notebooks/scrubbed (without updating from notebooks)
	poetry run python scripts/scrub_src.py notebooks notebooks/scrubbed

run_nb: FORCE  ## scrub and run passed notebook
	poetry run python scripts/run_nb.py notebooks notebooks/src $(XFILE)

.PHONY: help

.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

FORCE:
