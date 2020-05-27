Contribute to Ergo notebooks
============================

How to change a notebook and make a PR
--------------------------------------
1. Open the `notebook`_ in JupyterLab or Colab (:doc:`jupyter_colab`)
2. Make your changes
3. Follow our :doc:`notebook_style` 
4. Run the notebook in Colab. Save the .ipynb file (with output) in ``ergo/notebooks``
5. Run `make scrub`. This will produce a scrubbed version of the notebook in `ergo/notebooks/scrubbed`/.

   1. You can `git diff` the scrubbed version against the previous scrubbed version
   to more easily see what you changed
   
   2. You may want to use nbdime_ for better diffing

6. You can now make a PR with your changes. If you make a PR in the original ergo repo
(not a fork), you can then use the auto-comment from ReviewNB to more thoroughly vet your changes

.. _notebook: https://github.com/oughtinc/ergo/tree/master/notebooks
.. _nbdime: https://nbdime.readthedocs.io/en/latest/
