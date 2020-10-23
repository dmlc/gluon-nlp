#!/bin/bash

set -exs

SUBMIT_JOB_SCRIPT=../../tools/batch/submit-job.py

python3 benchmark_gluonnlp.py --layout NT --compute_layout NT --mode inference --use_tvm --instance_type g4

delare -A instance_job_type=( ["g4"]="g4dn.12x" ["p3"]="dog")

# Training speed on g4 and p3 instance
for instance in g4 p3
do
  for layouts in "NT NT" "NT TN" "TN TN"
  do
    read layout compute_layout <<< $ele
    python3 ${SUBMIT_JOB_SCRIPT} \
        --region us-east-1 \
        --source-ref master \
        --job-type g4dn.12x \
        --save-path temp \
        --name  \
        --work-dir scripts/benchmarks \
        --remote https://github.com/dmlc/gluon-nlp/ \
        --command "python3 benchmark_gluonnlp.py --layout NT --compute_layout NT --mode inference --use_tvm --instance_type g4 | tee stdout.log" >> ${LOG_PATH}
  done
done
