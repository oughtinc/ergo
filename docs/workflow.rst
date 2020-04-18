Workflow
========

Ergo uses the `Poetry`_ package manager.

Run a Colab using your local version of Ergo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This way, you can quickly make changes to ergo and see them reflected in
your Colab without pushing to a Github branch first.

1. ``poetry install``
2. ``poetry shell``
3. ``python -m jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com'``
4. Open the Colab in your browser. You need editing access to run the
   Colab – if you don’t, you can make a copy and run that instead.
5. In the Colab, ``Connect`` > ``Connect to local runtime``
6. For the ``Backend URL`` to connect to, paste from your shell the url
   that looks like
   “http://localhost:8888/?token=46aa5a3f5ee5b71df3c109fcabf94d0291b73bfced692049”
7. Whenever you change ``ergo`` and want to load the change in your
   Colab, in the Colab, ``Runtime`` > ``Restart Runtime...``

If you get an error in the Colab, try following the instructions
provided in the error. If that doesn’t work, try the `official
instructions for connecting to a local runtime`_.

Before submitting a PR
~~~~~~~~~~~~~~~~~~~~~~

1. Format code using ``make format`` (black, isort)
2. Run linting using ``make lint`` (flake8, mypy, black check)
3. Run tests using ``make test``
4. Generate docs using ``make docs``, review
   ``docs/build/html/index.html``

.. _Poetry: https://github.com/python-poetry/poetry
.. _official instructions for connecting to a local runtime: https://research.google.com/colaboratory/local-runtimes.html
