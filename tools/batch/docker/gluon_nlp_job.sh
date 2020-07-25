#!/bin/bash
date
echo "Args: $@"
env
echo "jobId: $AWS_BATCH_JOB_ID"
echo "jobQueue: $AWS_BATCH_JQ_NAME"
echo "computeEnvironment: $AWS_BATCH_CE_NAME"

SOURCE_REF=$1
WORK_DIR=$2
COMMAND=$3
SAVED_OUTPUT=$4
SAVE_PATH=$5
REMOTE=$6

if [ ! -z $REMOTE ]; then
    git remote set-url origin $REMOTE
fi;

git fetch origin $SOURCE_REF:working
git checkout working
pip3 install -v -e .[extras]
TIMESTAMP=$(date '+%Y%m%dT%H%M%SZ')

cd $WORK_DIR
/bin/bash -o pipefail -c "$COMMAND"
COMMAND_EXIT_CODE=$?
if [[ -f $SAVED_OUTPUT ]]; then
  aws s3 cp $SAVED_OUTPUT s3://gluon-nlp-dev/batch/$TIMESTAMP/$AWS_BATCH_JOB_ID/$SAVE_PATH;
elif [[ -d $SAVED_OUTPUT ]]; then
  aws s3 cp --recursive $SAVED_OUTPUT s3://gluon-nlp-dev/batch/$TIMESTAMP/$AWS_BATCH_JOB_ID/$SAVE_PATH;
fi;
exit $COMMAND_EXIT_CODE
