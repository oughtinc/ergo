# Ergo

A Python library for forecasting models

## Before submitting a PR
1. Format code according to [PEP8](https://www.python.org/dev/peps/pep-0008/). I use autopep8 (via the Python VSCode extension)
2. Run mypy: `mypy .`. There should be 0 errors or warnings, you should get `Success: no issues found`
3. Run tests: `pytest`. All tests should pass.
3. Increment the version of this package in:
    1. `pyproject.toml`
    2. `ergo/__init__.py`
    3. in tests as needed
4. Make sure the code works with our most recent Colabs:
    1. Push the code to a branch on Github
    2. Install the code in the colab like 

