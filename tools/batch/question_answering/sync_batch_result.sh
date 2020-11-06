#!/bin/bash

set -ex

LOG_PATH=$1
SAVE_DIR_NAME=${2:-squad_2.0}

while read -r job_name job_id; do
    aws s3 sync s3://gluon-nlp-dev/batch/${job_id}/temp ${SAVE_DIR_NAME}/${job_name}
done < ${LOG_PATH}
