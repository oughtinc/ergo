# Ergo

[![Build Status](https://travis-ci.org/oughtinc/ergo.svg?branch=master)](https://travis-ci.org/oughtinc/ergo)

A Python library for integrating model-based and judgmental forecasting

## Colabs using Ergo

- [Basic example](https://colab.research.google.com/github/oughtinc/ergo/blob/master/notebooks/basic.ipynb)
- [Models using Metaculus community distributions](https://colab.research.google.com/github/oughtinc/ergo/blob/community-prediction/notebooks/community_distributions.ipynb)

## Development
### Run a Colab using your local version of `ergo`
This way, you can quickly make changes to ergo and see them reflected in your Colab without pushing to a Github branch first.

1. `poetry shell`
2. `python -m jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com'`
3. open the Colab in your browser. You need editing access to run the Colab -- if you don't, you can make a copy and run that instead.
4. in the Colab, `Connect` > `Connect to local runtime`
5. for the `Backend URL` to connect to, paste from your shell the url that looks like “http://localhost:8888/?token=46aa5a3f5ee5b71df3c109fcabf94d0291b73bfced692049”
6. Whenever you change `ergo` and want to load the change in your Colab, in the Colab, `Runtime` > `Restart Runtime...`

If you get an error in the Colab, try following the instructions provided in the error. If that doesn't work, try the [official instructions for connecting to a local runtime](https://research.google.com/colaboratory/local-runtimes.html).