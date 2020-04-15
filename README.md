# Ergo

[![Build Status](https://travis-ci.org/oughtinc/ergo.svg?branch=master)](https://travis-ci.org/oughtinc/ergo)

A Python library for integrating model-based and judgmental forecasting

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
    - Submitting and monitoring questions to these platforms takes a long time
    - The questions on prediction platforms mostly aren't connected to decisions
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
- Some Covid-19-related data utils

WIP:

- Metaculus API improvements
  - Support questions with distributions on dates
  - Improve community distribution support (accuracy, plotting)
- Foretold API

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

2. [Relating Metaculus community distributions: Infections, Deaths, and IFR](notebooks/community-distributions.ipynb)
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
