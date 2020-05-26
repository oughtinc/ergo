Notebook contrib folder
=======================

Adding new packages
-------------------

For modules providing functionality specific to the questions
addressed in a notebook, create a new package in contrib
``/ergo/contrib/{your_package}`` and include an ``__init__.py``
file. You can then access it in your notebook with:

.. code-block:: python

   from ergo.contrib.{your_package} import {module_you_want}

For modules providing more general functionality of use across
notebooks (and perhaps a candidate for inclusion in core ergo), you
can use ``/ergo/contrib/utils``. You can either add a new module or
extend an existing one. You can then access it with:

.. code-block:: python

   from ergo.contrib.utils import {module_you_want}

Adding dependencies
-------------------

1. Usual poetry way with --optional flag

.. code-block:: bash

   poetry add {pendulum} --optional

2. You can then (manually in the ``pyproject.toml``) add it to the
   'notebook' group

(Look for "extras" in ``pyproject.toml``)

.. code-block:: toml
                
   [tool.poetry.extras]
   notebooks = [   
                "pendulum",
                "scikit-learn",
                "{your_dependency}"
               ]                        
   

(To my knowledge) there is no way currently to do this second step
with the CLI.

This allows people to then install the additional
notebook dependencies with:  
.. code-block:: bash

   poetry install -E notebooks
