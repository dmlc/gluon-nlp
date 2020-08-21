#!/bin/bash
# Shell script for installing dependencies and running test on AWS Batch

# alias python3='/usr/bin/python3'

sudo apt-get install libopenblas-dev
python3 -m pip install --user -upgrade pip
python3 -m pip install --user setuptools pytest pytest-cov contextvars
python3 -m pip install --upgrade cython
python3 -m pip install --pre --user "mxnet-cu102>=2.0.0b20200802" -f https://dist.mxnet.io/python
python3 -m pip install --user -e .[extras]
python3 -m pytest --cov=./ --cov-report=xml --durations=50 --device="mx.gpu()" tests/
