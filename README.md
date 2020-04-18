[![Build Status](https://travis-ci.org/oughtinc/ergo.svg?branch=master)](https://travis-ci.org/oughtinc/ergo) [![Documentation Status](https://readthedocs.com/projects/ought-ergo/badge/?version=latest&token=259162a0cd579e231ba0828410ff8b8f813f5eac663dcc8882b1244decdc97ae)](https://ought-ergo.readthedocs-hosted.com/en/latest/?badge=latest) [![Codecov Status](https://codecov.io/gh/oughtinc/ergo/branch/master/graph/badge.svg)](https://codecov.io/gh/oughtinc/ergo)

# Ergo

A Python library for integrating model-based and judgmental forecasting

[Quickstart](#quickstart) | [Docs](https://ergo.ought.org) | [Examples](#notebooks-using-ergo)

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

## Quickstart

1. Open [this Colab](https://colab.research.google.com/github/oughtinc/ergo/blob/master/notebooks/quickstart.ipynb)
2. Add your Metaculus username and password
3. Select "Runtime > Run all" in the menu
4. Edit the code to load other questions, improve the model, etc., and rerun

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
    - Very few people build models
    - Very few people submit questions to prediction platforms, or predict on these platforms
    - Improvements to forecasting accrue slowly
    - Most decisions are not informed by systematic forecasts
4. Better infrastructure for forecasting can connect the pieces and help realize the potential of scalable high-quality forecasting


## Functionality

Ergo is still at a very early stage. Functionality and API are in flux.

Here's what Ergo provides right now:

- Express generative models in a probabilistic programming language
  - Ergo provides lightweight wrappers around [Pyro](https://pyro.ai) functions to make the models more readable
  - Specify distributions using 90% confidence intervals, e.g. `ergo.lognormal_from_interval(10, 100)`
  - For Bayesian inference, Ergo provides a wrapper around Pyro's variational inference algorithm
  - Get model results as Pandas dataframes
- Interact with the Metaculus prediction platform
  - Load question data given question ids
  - Use community distributions as variables in generative models
  - Submit model predictions to Metaculus
    - We automatically fit a mixture of logistic distributions for continuous-valued questions
  - Plot community distributions
- Load question data (distributions) from [Foretold](https://www.foretold.io/) prediction platform
- Some Covid-19-related data utils

[WIP](https://github.com/oughtinc/ergo/projects/1):

- Metaculus API improvements
  - Support questions with distributions on dates
  - Improve community distribution support (accuracy, plotting)
- Foretold API improvements
  - Submitting data

Planned:

- Interfaces for all prediction platforms
  - Search questions on prediction platforms
  - Use distributions from any platform
  - Programmatically submit questions to platforms
  - Track community distribution changes
- Frequently used model components
  - Index/ensemble models that summarize fuzzy large questions like "What's going to happen with the economy next year?"
  - Model components for integrating qualitative adjustments into quantitative models
  - Simple probability decomposition models
  - E.g. see [The Model Thinker](https://www.amazon.com/Model-Thinker-What-Need-Know/dp/0465094627) (Scott Page)
- Better tools for integrating models and platforms
  - Compute model-based predictions by constraining model variables to be close to the community distributions
- Push/pull to and from repositories for generative models
  - Think [Forest](http://forestdb.org) + Github

If there's something you want Ergo to do, [let us know](https://github.com/oughtinc/ergo/issues)!

## Notebooks using Ergo

The following notebooks have been created at different points in time and use Ergo in inconsistent ways. Most are rough scratchpads of work-in-progress and haven't been cleaned up for public consumption:

1. [Generative models in Ergo](notebooks/generative-models.ipynb)
    - Models in Ergo are expressed as probabilistic programs. 
    - This notebook shows a simple example. 
    - As in [Guesstimate](https://www.getguesstimate.com), you can define distributions from 90% confidence intervals.

2. [Relating Metaculus community distributions: Infections, Deaths, and IFR](notebooks/community_distributions_v2.ipynb)
    - A notebook for the model shown above that uses a model to update Metaculus community distributions towards consistency

3. [Model-based predictions of Covid-19 spread](notebooks/covid-19-metaculus.ipynb)
   - End-to-end example: 
     1. Load multiple questions from Metaculus
     2. Compute model predictions based on assumptions and external data
     3. Submit predictions to Metaculus

4. [Model-based predictions of Covid-19 spread using inference from observed cases](notebooks/covid-19-inference.ipynb)
   - A version of the previous notebook that infers growth rates before and after lockdown decisions

5. [Predicting how long lockdowns will last in multiple locations](notebooks/covid-19-lockdowns.ipynb) (WIP)
   - Make predictions on multiple Metaculus questions using external data (IHME) and a single model.

6. [Estimating the number of active Covid-19 infections in each country using multiple sources](notebooks/covid-19-active.ipynb) (WIP)
   - Integrate qualitative judgments and simple model-based extrapolation to estimate the number of active cases for a large number of countries.
   
7. [How long will the average American spend under lockdown?](notebooks/covid-19-average-lockdown.ipynb) (WIP)
   - Show how related questions and how their community prediction has changed since making a prediction.
   
8. [Assorted predictions](notebooks/assorted-predictions.ipynb)
   - Nine quick predictions

9. [Prediction dashboard](notebooks/prediction-dashboard.ipynb)
   - Show Metaculus prediction results as a dataframe
   - Filter Metaculus questions by date and status.

   

Notebooks on the path to Ergo (hosted on Colab):

1. [Guesstimate in Colab](https://colab.research.google.com/drive/1V9eR6T1RAbtfpZYFaueL8miBJ6wgLXIm)
   - How can we get Guesstimate's sampling and visualization functionality in a Colab?
   
2. [Fitting mixtures of logistic distributions](https://colab.research.google.com/drive/1xwO-0A36wnut9GPlEaRj6zzZBBLf1T2C)
   - How can we transform arbitrary distributions represented as samples into the "mixtures of logistics" format Metaculus uses for user submissions?

## Development

Ergo uses the [Poetry](https://github.com/python-poetry/poetry) package manager.

### Run a Colab using your local version of `ergo`

This way, you can quickly make changes to ergo and see them reflected in your Colab without pushing to a Github branch first.

1. `poetry install`
2. `poetry shell`
3. `python -m jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com'`
4. Open the Colab in your browser. You need editing access to run the Colab -- if you don't, you can make a copy and run that instead.
5. In the Colab, `Connect` > `Connect to local runtime`
6. For the `Backend URL` to connect to, paste from your shell the url that looks like “http://localhost:8888/?token=46aa5a3f5ee5b71df3c109fcabf94d0291b73bfced692049”
7. Whenever you change `ergo` and want to load the change in your Colab, in the Colab, `Runtime` > `Restart Runtime...`

If you get an error in the Colab, try following the instructions provided in the error. If that doesn't work, try the [official instructions for connecting to a local runtime](https://research.google.com/colaboratory/local-runtimes.html).

### Before submitting a PR

1. Format code using [black](https://github.com/psf/black).
    * Follow [these instructions](https://code.visualstudio.com/docs/python/editing#_formatting) to set `black` as your formatter in VSCode.
2. Lint code using [flake8](https://flake8.pycqa.org/en/latest/).
    * Follow [these instructions](https://code.visualstudio.com/docs/python/linting#_specific-linters) to use flake8 in VSCode
3. Run mypy: `mypy .`. There should be 0 errors or warnings, you should get `Success: no issues found`
4. Run tests: `pytest -s`. All tests should pass.
