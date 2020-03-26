# Ergo

A Python library for forecasting models

## Before submitting a PR

1. Format code according to [PEP8](https://www.python.org/dev/peps/pep-0008/). You could use autopep8.
2. Run mypy: `mypy .`. There should be 0 errors or warnings, you should get `Success: no issues found`
3. Run tests: `pytest -s`. All tests should pass.
3. Increment the version of this package in:
    1. `pyproject.toml`
    2. `ergo/__init__.py`
    3. In tests as needed
4. Make sure the code works with our most recent Colabs:
    1. Push the code to a branch on Github
    2. Install the code in the Colab like `!pip install --quiet git+https://github.com/oughtinc/ergo.git@branch-name`
        1. Note that if you make changes to the library and need to reload it into the Colab, you'll need to first go to `Runtime > Factory reset runtime` in the Colab (`Restart runtime` will not cause the Colab to reload the library from Github)
    3. Run the Colab, make sure it works. Make sure not to submit any predictions on our main account, use our test credentials if necessary: Username: `oughttest`, Password: `6vCo39Mz^rrb`
