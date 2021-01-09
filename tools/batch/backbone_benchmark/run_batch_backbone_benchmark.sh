#!/bin/bash

set -ex
LOG_PATH=${1:-submit_backbone_benchmark.log}
INSTANCE_TYPE=${2:-g4dn.2x}
SUBMIT_SCRIPT_PATH=$(dirname "$0")/../../../tools/batch/submit-job.py

python3 ${SUBMIT_SCRIPT_PATH} \
    --region us-east-1 \
    --source-ref fix_benchmark3 \
    --job-type ${INSTANCE_TYPE} \
    --save-path temp \
    --name test_backbone_benchmark \
    --work-dir scripts/benchmarks \
    --remote https://github.com/sxjscience/gluon-nlp/ \
    --command "bash run_backbone_bench_${INSTANCE_TYPE}.sh | tee stdout.log" \
    | perl -pe 's/Submitted job \[([0-9|a-z|_].+)\] to the job queue .+/$1/' \
    | sed -e 's/ - / /g' >> ${LOG_PATH}
