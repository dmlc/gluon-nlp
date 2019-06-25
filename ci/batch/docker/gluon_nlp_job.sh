#!/bin/bash
date
echo "Args: $@"
env
echo "jobId: $AWS_BATCH_JOB_ID"
echo "jobQueue: $AWS_BATCH_JQ_NAME"
echo "computeEnvironment: $AWS_BATCH_CE_NAME"

SOURCE_REF=$1
CONDA_ENV=$2
WORK_DIR=$3
COMMAND=$4
SAVED_OUTPUT=$5
SAVE_PATH=$6
REMOTE=$7

if [ ! -z $REMOTE ]; then
    git remote set-url origin $REMOTE
fi;

git fetch origin $SOURCE_REF:working
git checkout working
conda env update --prune -p conda/$CONDA_ENV -f env/$CONDA_ENV.yml
source activate ./conda/$CONDA_ENV
pip install -v -e .
python -m spacy download en
python -m spacy download de
python -m nltk.downloader all
pip install awscli

cd $WORK_DIR
/bin/bash -c "$COMMAND"
COMMAND_EXIT_CODE=$?
if [[ -f $SAVED_OUTPUT ]]; then
  aws s3 cp $SAVED_OUTPUT s3://gluon-nlp-staging/$SAVE_PATH;
elif [[ -d $SAVED_OUTPUT ]]; then
  aws s3 cp --recursive $SAVED_OUTPUT s3://gluon-nlp-staging/$SAVE_PATH;
fi;
exit $COMMAND_EXIT_CODE
