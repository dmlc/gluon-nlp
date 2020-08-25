#!/bin/bash
# Shell script for installing dependencies and running test on AWS Batch

# alias python3='/usr/bin/python3'

echo $PWD

sudo apt-get install libopenblas-dev
python3 -m pip install --user -upgrade pip
python3 -m pip install --user setuptools pytest pytest-cov contextvars
python3 -m pip install --upgrade cython
python3 -m pip install --pre --user "mxnet-cu102>=2.0.0b20200802" -f https://dist.mxnet.io/python
python3 -m pip install --user -e .[extras]
python3 -m pytest --cov=./ --cov-report=xml --durations=50 --device="gpu" ../../tests/ > output.txt

flag=false
while IFS= read -r line; do
    if $flag; then
        echo $line
    else
        if [ "$line" == "/gluon-nlp/tools/batch" ]; then
            echo $line
            flag=true
        fi
    fi
done < output.txt
