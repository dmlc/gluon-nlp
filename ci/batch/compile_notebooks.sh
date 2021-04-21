#!/bin/bash
# Shell script for submitting AWS Batch jobs to compile notebooks
event=$1
ref=$2

# Skip reason: https://github.com/apache/incubator-mxnet/issues/20197
SkipCompute=("word_embedding_training.md")

containFile () {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

compile_notebook () {
    local MDFILE=$1
    DIR=$(dirname $MDFILE)
    BASENAME=$(basename $MDFILE)
    TARGETNAME=$(dirname $MDFILE)/${BASENAME%.md}.ipynb

    echo Compiling $BASENAME ...
    
    containFile $BASENAME "${SkipCompute[@]}"
    FIND=$?
    if [ $FIND -eq 0 ]; then
        python3 docs/md2ipynb.py -d ${MDFILE}
    else
        python3 docs/md2ipynb.py ${MDFILE}
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo Compiling $BASENAME Failed
        exit $EXIT_CODE
    else
        echo Compiling $BASENAME Succeeded
    fi
}

for f in $(find docs/examples -type f -name '*.md' -print); do
    compile_notebook "$f"
done;

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo Building Website
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
else
    exit 1
fi
