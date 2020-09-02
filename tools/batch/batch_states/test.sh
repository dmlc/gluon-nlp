#!/bin/bash
# Shell script for installing dependencies and running test on AWS Batch

echo $PWD

python3 -m pip install --user --quiet -upgrade pip
python3 -m pip install --user --quiet setuptools pytest pytest-cov contextvars
python3 -m pip install --upgrade --quiet cython
python3 -m pytest --cov=./ --cov-report=xml --durations=50 --device="gpu" --runslow /gluon-nlp/tests/
