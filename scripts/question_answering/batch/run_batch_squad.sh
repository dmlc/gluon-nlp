#!/bin/bash

set -ex

USE_HOROVOD=${1:-0}
VERSION=${2:-2.0}
LOG_PATH=${3:-submit_squad_v2.log}
DTYPE=${4:-float32}
SUBMIT_SCRIPT_PATH=$(dirname "$0")/../../../tools/batch/submit-job.py


for MODEL_NAME in albert_base \
                  albert_large \
                  albert_xlarge \
                  albert_xxlarge \
                  electra_base \
                  electra_large \
                  electra_small \
                  roberta_large \
                  uncased_bert_base \
                  uncased_bert_large \
                  gluon_en_cased_bert_base_v1 \
                  mobilebert
do
  python3 ${SUBMIT_SCRIPT_PATH} \
      --region us-east-1 \
      --source-ref amp \
      --job-type g4dn.12x \
      --save-path temp \
      --name test_squad2_${MODEL_NAME} \
      --work-dir scripts/question_answering \
      --remote https://github.com/sxjscience/gluon-nlp/ \
      --command "bash commands/run_squad2_${MODEL_NAME}.sh ${USE_HOROVOD} ${VERSION} ${DTYPE} | tee stdout.log" \
      | perl -pe 's/Submitted job \[([0-9|a-z|_].+)\] to the job queue .+/$1/' \
      | sed -e 's/ - / /g' >> ${LOG_PATH}
done
