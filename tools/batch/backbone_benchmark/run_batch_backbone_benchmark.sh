#!/bin/bash

set -ex
INSTANCE_TYPE=${1:-g4dn.2x}
LOG_PATH=${2:-submit_backbone_benchmark.log}
SUBMIT_SCRIPT_PATH=$(dirname "$0")/../../../tools/batch/submit-job.py

python3 ${SUBMIT_SCRIPT_PATH} \
    --region us-east-1 \
    --source-ref fix_benchmark3 \
    --job-type ${INSTANCE_TYPE} \
    --save-path temp \
    --name test_backbone_benchmark_${INSTANCE_TYPE} \
    --work-dir scripts/benchmarks \
    --remote https://github.com/sxjscience/gluon-nlp/ \
    --command "bash run_backbone_benchmark.sh 2>&1 | tee stdout.log" \
    | perl -pe 's/Submitted job \[([0-9|a-z|_].+)\] to the job queue .+/$1/' \
    | sed -e 's/ - / /g' >> ${LOG_PATH}
