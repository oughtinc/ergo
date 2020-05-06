Getting Started
===============

Try Ergo
--------

The quickest way to try Ergo right now is by starting with a template Colab:

1. Open `this Colab`_
2. Add your Metaculus username and password
3. Select “Runtime > Run all” in the menu
4. Edit the code to load other questions, improve the model, etc., and
   rerun

.. _this Colab: https://colab.research.google.com/github/oughtinc/ergo/blob/master/notebooks/quickstart.ipynb

Installation
------------

If you want to use Ergo locally, you can install it using ``pip install git+https://github.com/oughtinc/ergo.git``.

How to use
----------

There isn't much documentation for Ergo yet.

For now, take a look at some of our colab notebooks using Ergo:

1. `Relating Metaculus community distributions: Infections, Deaths, and
   IFR`_

   -  A notebook for the model shown above that uses a model to update
      Metaculus community distributions towards consistency

2. `Model-based predictions of Covid-19 spread`_

   -  End-to-end example:

      1. Load multiple questions from Metaculus
      2. Compute model predictions based on assumptions and external
         data
      3. Submit predictions to Metaculus

3. `Model-based predictions of Covid-19 spread using inference from
   observed cases`_

   -  A version of the previous notebook that infers growth rates before
      and after lockdown decisions

4. `Prediction dashboard`_

   -  Show Metaculus prediction results as a dataframe
   -  Filter Metaculus questions by date and status.

.. _`Relating Metaculus community distributions: Infections, Deaths, and IFR`: https://github.com/oughtinc/ergo/tree/master/notebooks/build/community-distributions.ipynb
.. _Model-based predictions of Covid-19 spread: https://github.com/oughtinc/ergo/tree/master/notebooks/build/covid-19-metaculus.ipynb
.. _Model-based predictions of Covid-19 spread using inference from observed cases: https://github.com/oughtinc/ergo/tree/master/notebooks/build/covid-19-inference.ipynb
.. _Prediction dashboard: https://github.com/oughtinc/ergo/tree/master/notebooks/build/prediction-dashboard.ipynb

These notebooks have been created at different points in time and use Ergo in inconsistent ways.
