#!/usr/bin/env python
from setuptools import find_packages
from distutils.core import setup

setup(
    name="ergo",
    version="0.8.3",
    description="A Python library for integrating model-based and judgmental forecasting",
    author="Ought",
    author_email="ergo@ought.org",
    url="https://ought.org",
    packages=find_packages(),
)
