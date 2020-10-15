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
DEVICE=${7:-gpu}

if [ ! -z $REMOTE ]; then
    git remote set-url origin $REMOTE
fi;

git fetch origin $SOURCE_REF:working
git checkout working

if [ $DEVICE == "cpu" ]; then
  python3 -m pip install -U --quiet --pre "mxnet>=2.0.0b20200802" -f https://dist.mxnet.io/python
else
  # Due to the issue in https://forums.aws.amazon.com/thread.jspa?messageID=953912
  # We need to manually configure the shm to ensure that Horovod is runnable.
  # The reason that we need a larger shm is described in https://github.com/NVIDIA/nccl/issues/290
  umount shm
  mount -t tmpfs -o rw,nosuid,nodev,noexec,relatime,size=2G shm /dev/shm
  python3 -m pip install -U --quiet --pre "mxnet-cu102>=2.0.0b20200802" -f https://dist.mxnet.io/python
fi

python3 -m pip install --quiet -e .[extras]

cd $WORK_DIR
/bin/bash -o pipefail -c "$COMMAND"
COMMAND_EXIT_CODE=$?
if [[ -f $SAVED_OUTPUT ]]; then
  aws s3 cp $SAVED_OUTPUT s3://gluon-nlp-dev/batch/$AWS_BATCH_JOB_ID/$SAVE_PATH;
elif [[ -d $SAVED_OUTPUT ]]; then
  aws s3 cp --recursive $SAVED_OUTPUT s3://gluon-nlp-dev/batch/$AWS_BATCH_JOB_ID/$SAVE_PATH;
fi;
exit $COMMAND_EXIT_CODE
