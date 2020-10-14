#!/bin/bash
# Shell script for installing dependencies and running test on AWS Batch
set -ex

echo $PWD
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
REPODIR=${SCRIPTPATH}/../../../

python3 -m pytest --cov=$REPODIR --cov-config=$REPODIR/.coveragerc --cov-report=xml --durations=50 --device="gpu" --runslow $REPODIR/tests/
