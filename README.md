[![Build Status](https://travis-ci.org/oughtinc/ergo.svg?branch=master)](https://travis-ci.org/oughtinc/ergo) [![Documentation Status](https://readthedocs.org/projects/ergo/badge/?version=latest)](https://ergo.ought.org/) [![Codecov Status](https://codecov.io/gh/oughtinc/ergo/branch/master/graph/badge.svg)](https://codecov.io/gh/oughtinc/ergo)

# Ergo

A Python library for integrating model-based and judgmental forecasting

[Quickstart](#get-started-using-ergo) | [Docs](https://ergo.ought.org) | [Examples](#notebooks-using-ergo)

## Example

We'll relate three questions on the [Metaculus](https://www.metaculus.com) crowd prediction platform using a generative model:

```py
# Log into Metaculus
metaculus = ergo.Metaculus(username="ought", password="")

# Load three questions
q_infections = metaculus.get_question(3529, name="Covid-19 infections in 2020")
q_deaths = metaculus.get_question(3530, name="Covid-19 deaths in 2020")
q_ratio = metaculus.get_question(3755, name="Covid-19 ratio of fatalities to infections")

# Relate the three questions using a generative model
def deaths_from_infections():
    infections = q_infections.sample_community()
    ratio = q_ratio.sample_community()
    deaths = infections * ratio
    ergo.tag(deaths, "Covid-19 deaths in 2020")
    return deaths

# Compute model predictions for the `deaths` question
samples = ergo.run(deaths_from_infections, num_samples=5000)

# Submit model predictions to Metaculus
q_deaths.submit_from_samples(samples)
```

You can run the model [here](https://colab.research.google.com/github/oughtinc/ergo/blob/master/notebooks/community-distributions.ipynb).

## Get started using Ergo

1. Open [this Colab](https://colab.research.google.com/github/oughtinc/ergo/blob/master/notebooks/quickstart.ipynb)
2. Add your Metaculus username and password
3. Select "Runtime > Run all" in the menu
4. Edit the code to load other questions, improve the model, etc., and rerun

## Notebooks using Ergo

This notebook is closest to a tutorial right now:

- [El Paso workflow](notebooks/el-paso-workflow.ipynb)
  - This notebook shows multi-level decomposition, Metaculus community distributions, ensembling, and beta-binomial and log-normal distributions using part of the [El Paso Covid-19 model](notebooks/el-paso.ipynb).

The notebooks below have been created at different points in time and use Ergo in inconsistent ways. Most are rough scratchpads of work-in-progress and haven't been cleaned up for public consumption:

1. [Relating Metaculus community distributions: Infections, Deaths, and IFR](notebooks/community-distributions.ipynb)

   - A notebook for the model shown above that uses a model to update Metaculus community distributions towards consistency

2. [Model-based predictions of Covid-19 spread](notebooks/covid-19-metaculus.ipynb)

   - End-to-end example:
     1. Load multiple questions from Metaculus
     2. Compute model predictions based on assumptions and external data
     3. Submit predictions to Metaculus

3. [Model-based predictions of Covid-19 spread using inference from observed cases](notebooks/covid-19-inference.ipynb)

   - A version of the previous notebook that infers growth rates before and after lockdown decisions

4. [Metaculus questions data](notebooks/metac_qs_data.ipynb)

- Get rich metadata on open Metaculus questions

5. [Prediction dashboard](notebooks/prediction-dashboard.ipynb)

   - Show Metaculus prediction results as a dataframe
   - Filter Metaculus questions by date and status.

6. [El Paso questions](notebooks/el-paso.ipynb)
   - Illustrates how to load all questions for a Metaculus category (in this case for the [El Paso series](https://pandemic.metaculus.com/questions/4161/el-paso-series-supporting-covid-19-response-planning-in-a-mid-sized-city/))

Outdated Ergo notebooks:

1. [Generative models in Ergo](notebooks/generative-models.ipynb)

2. [Predicting how long lockdowns will last in multiple locations](notebooks/covid-19-lockdowns.ipynb)

3. [Estimating the number of active Covid-19 infections in each country using multiple sources](notebooks/covid-19-active.ipynb)

4. [How long will the average American spend under lockdown?](notebooks/covid-19-average-lockdown.ipynb)

5. [Assorted COVID predictions](notebooks/assorted-predictions.ipynb)

## Local installation

To install Ergo and its dependencies, we recommend PyEnv and Poetry:

1. Install [PyEnv](https://github.com/pyenv/pyenv-installer) for managing Python versions
2. Install the [Poetry](https://python-poetry.org/docs/) package manager

Then:

```
mkdir my-ergo-project && cd my-ergo-project
pyenv install 3.6.9 && pyenv local 3.6.9
poetry init -n
# Edit pyproject.toml to set python = "~3.6.9"
poetry add git+https://github.com/oughtinc/ergo.git
poetry install
```

Now Ergo is available in your project:

```
poetry run python
>>> import ergo
>>> ergo.flip(.5)
DeviceArray(True, dtype=bool)
```

## Contribute

Ergo is an open source project and we love contributions!

See our [instructions for contributors](CONTRIBUTING.md) for more.

## Philosophy

The theory behind Ergo:

1. Many of the pieces necessary for good forecasting work are out there:
   - Prediction platforms
   - Probabilistic programming languages
   - Superforecasters + qualitative human judgments
   - Data science tools like numpy and pandas
   - Deep neural nets as expressive function approximators
2. But they haven't been connected yet in a productive workflow:
   - It's difficult to get data in and out of prediction platforms
   - Submitting questions to these platforms takes a long time
   - The questions on prediction platforms aren't connected to decisions, or even to other questions on the same platform
   - Human judgments don't scale
   - Models often can't take into account all relevant considerations
   - Workflows aren't made explicit so they can't be automated
3. This limits their potential:
   - Few people build models
   - Few people submit questions to prediction platforms, or predict on these platforms
   - Improvements to forecasting accrue slowly
   - Most decisions are not informed by systematic forecasts
4. Better infrastructure for forecasting can connect the pieces and help realize the potential of scalable high-quality forecasting

## Functionality

Ergo is still at an early stage. Pre-alpha, or whatever the earliest possible stage is. Functionality and API are in flux.

Here's what Ergo provides right now:

- Express generative models in a probabilistic programming language
  - Ergo provides lightweight wrappers around [Pyro](https://pyro.ai) functions to make the models more readable
  - Specify distributions using 90% confidence intervals, e.g. `ergo.lognormal_from_interval(10, 100)`
  - For Bayesian inference, Ergo provides a wrapper around Pyro's variational inference algorithm
  - Get model results as Pandas dataframes
- Interact with the [Metaculus](https://www.metaculus.com/) and [Foretold](https://www.foretold.io/) prediction platforms
  - Load question data given question ids
  - Use community distributions as variables in generative models
  - Submit model predictions to these platforms
    - For Metaculus, we automatically fit a mixture of logistic distributions for continuous-valued questions
  - Plot community distributions

[WIP](https://github.com/oughtinc/ergo/projects/1):

- Documentation
- Clearer modeling API

Planned:

- Interfaces for all prediction platforms
  - Search questions on prediction platforms
  - Use distributions from any platform
  - Programmatically submit questions to platforms
  - Track community distribution changes
- Common model components
  - Index/ensemble models that summarize fuzzy large questions like "What's going to happen with the economy next year?"
  - Model components for integrating qualitative adjustments into quantitative models
  - Simple probability decomposition models
  - E.g. see [The Model Thinker](https://www.amazon.com/Model-Thinker-What-Need-Know/dp/0465094627) (Scott Page)
- Better tools for integrating models and platforms
  - Compute model-based predictions by constraining model variables to be close to the community distributions
- Push/pull to and from repositories for generative models
  - Think [Forest](http://forestdb.org) + Github

If there's something you want Ergo to do, [let us know](https://github.com/oughtinc/ergo/issues)!
