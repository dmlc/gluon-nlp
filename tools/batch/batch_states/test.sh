#!/bin/bash
# Shell script for installing dependencies and running test on AWS Batch
set -ex

echo $PWD
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
REPODIR="$( readlink -f ${SCRIPTPATH}/../../../../gluon-nlp)"

python3 -m pip install --upgrade --user pytest pytest-cov contextvars
python3 -m pytest --cov=$REPODIR --cov-config=$REPODIR/.coveragerc --cov-report=xml --durations=50 --device="gpu" --runslow $REPODIR/tests/
