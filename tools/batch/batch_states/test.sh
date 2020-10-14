#!/bin/bash
# Shell script for installing dependencies and running test on AWS Batch
set -ex

echo $PWD
DIRNAME=$(dirname $0)
REPODIR=$DIRNAME/../../../

python3 -m pytest --cov=$REPODIR --cov-config=$REPODIR/.coveragerc --cov-report=xml --durations=50 --device="gpu" --runslow $REPODIR/tests/
