Notebook Style
==============

How to clean up a notebook for us to feature:

1. Make sure that the notebook meets a high standard in general:

   1. high-quality code
   2. illuminating data analysis
   3. clear communication of what you’re doing and your findings
   4. as short as possible, but no shorter
   5. this `random style guide`_ I found in a few minutes of Googling
      seems good, but it’s not our official style guide or anything


2. Do the following specific things to clean up:
   
   1. as much as possible, avoid showing extraneous output from cells
      
      1. you can use the ``%%capture`` magic to suppress all output
         from a cell (helpful if a function in the cell prints
         something)
      2. you can add a ``;`` at the end of the last line in a cell to
         suppress printing the return value of the line
      3. think about what cells the reader really needs to see
         vs. which ones just have to be there for setup or
         whatnot. Collapse the latter.
      
   3. use the latest version of ``ergo``
   4. make sure that any secrets like passwords are removed from the
      notebook
   5. Pull out any code not central to the main point of the model
      into a module in ``ergo/contrib/``. See :doc:`contrib_folder` for
      details. 

The featured notebooks in our README should be exemplars of the
above, so refer to those to see what this looks like in practice.

.. _random style guide: https://github.com/spacetelescope/style-guides/blob/master/guides/jupyter-notebooks.md
.. _El Paso COVID predictions notebook: https://github.com/oughtinc/ergo/blob/master/notebooks/el-paso.ipynb
