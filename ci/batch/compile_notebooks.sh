#!/bin/bash
# Shell script for submitting AWS Batch jobs to compile notebooks

event=$1
ref=$2

make docs
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo Building Website Failed.
    exit $EXIT_CODE
else
    echo Building Website Succeeded.
    if [ "$1" == "push" ]; then
        echo "Uploading docs to s3://gluon-nlp/$2/"
        aws s3 sync --delete ./docs/_build/html/ s3://gluon-nlp/$2/ --quiet --acl public-read
    else
        echo "Uploading docs to s3://gluon-nlp-staging/PR$1/$2/"
        aws s3 sync --delete ./docs/_build/html/ s3://gluon-nlp-staging/PR$1/$2/ --quiet --acl public-read
    fi
fi