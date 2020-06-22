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
pip install -v -e .[extras]

cd $WORK_DIR
/bin/bash -o pipefail -c "$COMMAND"
COMMAND_EXIT_CODE=$?
if [[ -f $SAVED_OUTPUT ]]; then
  aws s3 cp $SAVED_OUTPUT s3://gluon-nlp-staging/$SAVE_PATH;
elif [[ -d $SAVED_OUTPUT ]]; then
  aws s3 cp --recursive $SAVED_OUTPUT s3://gluon-nlp-staging/$SAVE_PATH;
fi;
exit $COMMAND_EXIT_CODE
