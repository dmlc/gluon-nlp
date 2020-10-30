#!/bin/bash
# Shell script for submitting AWS Batch jobs to compile notebooks

prnumber=$1
runnumber=$2
remote=$3
refs=$4

FAIL=0

compile_notebook () {
    local MDFILE=$1
    DIR=$(dirname $MDFILE)
    BASENAME=$(basename $MDFILE)
    TARGETNAME=$(dirname $MDFILE)/${BASENAME%.md}.ipynb
    LOGNAME=${BASENAME%.md}.stdout.log
    JOBIDLOG=${BASENAME%.md}.jobid.log

    echo Compiling $BASENAME ...

    python3 tools/batch/submit-job.py --region us-east-1 \
            --wait \
            --timeout 3600 \
            --saved-output docs/examples \
            --name GluonNLP-Docs-${refs}-${prnumber}-${runnumber} \
            --save-path ${runnumber}/gluon-nlp/docs/examples \
            --work-dir . \
            --source-ref ${refs} \
            --remote https://github.com/${remote} \
            --command "python3 -m pip install --quiet nbformat notedown jupyter_client ipykernel \
                       ipykernel matplotlib termcolor && \
                       python3 docs/md2ipynb.py ${MDFILE}" > $LOGNAME

    BATCH_EXIT_CODE=$?

    head -100 $LOGNAME | grep -oP -m 1 'jobId: \K(.*)' > $JOBIDLOG

    JOBID=$(cat "$JOBIDLOG")

    if [ $BATCH_EXIT_CODE -ne 0 ]; then
        echo Compiling $BASENAME Failed, please download Notebook_Logs in build Artifacts for more details.
    else
        echo Compiling $BASENAME Succeeded
        aws s3api wait object-exists --bucket gluon-nlp-dev \
            --key batch/$JOBID/${runnumber}/gluon-nlp/$TARGETNAME
        aws s3 cp s3://gluon-nlp-dev/batch/$JOBID/${runnumber}/gluon-nlp/$TARGETNAME $TARGETNAME --quiet
    fi
    exit $BATCH_EXIT_CODE
}

pids=()

for f in $(find docs/examples -type f -name '*.md' -print); do
    compile_notebook "$f" &
    pids+=($!)
done;

for pid in "${pids[@]}"; do
    wait "$pid" || let "FAIL+=1"
done;

if [ "$FAIL" == "0" ]; then
    exit 0
else
    exit 1
fi
