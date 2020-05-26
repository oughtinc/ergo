Contribute to ergo core
=======================

To get started:

1. ``git clone https://github.com/oughtinc/ergo.git``
2. ``poetry install``
3. ``poetry shell``

``poetry``
----------
Ergo uses poetry to manage its dependencies and environments.

Folow these_ directions to install poetry if you don't already have it.

Troubleshooting: If you get `Could not find a version that satisfies the requirement jaxlib ...` after using poetry to install, this is probably because your virtual environment has old version of pip due to how poetry choses pip versions_.
Try `poetry run pip install -U pip`, then try `poetry install` agai

.. _these: https://python-poetry.org/docs/#installation
.. _versions: https://github.com/python-poetry/poetry/issues/732

Before submitting a PR
~~~~~~~~~~~~~~~~~~~~~~

1. Run ``poetry install`` to make sure you have the latest dependencies
2. Format code using ``make format`` (black, isort)
3. Run linting using ``make lint`` (flake8, mypy, black check)
4. Run tests using ``make test``

    * To run the tests in ``test_metaculus.py``, you'll need the password to
      our ``oughttest`` account. If you don't have it, you can ask us for it, 
      or rely on Travis CI to run those tests for you.

5. Generate docs using ``make docs``, load
   ``docs/build/html/index.html`` and review the generated docs
6. Or run all of the above using ``make all``
   
.. _Poetry: https://github.com/python-poetry/poetry
.. _official instructions for connecting to a local runtime: https://research.google.com/colaboratory/local-runtimes.html

Conventions
~~~~~~~~~~~

Import ``numpy`` as follows:


.. code-block:: python

    import jax.numpy as np
    import numpy as onp 

