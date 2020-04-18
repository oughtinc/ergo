lint: FORCE
	poetry run flake8
	poetry run mypy .
	poetry run black . --check

test: FORCE
	poetry run python -m pytest --cov=ergo -s .

format: FORCE
	poetry run isort -rc .
	poetry run black .

docs: FORCE
	poetry run $(MAKE) -C docs html

FORCE:
