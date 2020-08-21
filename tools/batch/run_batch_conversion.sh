MODEL_NAME=$1

python3 tools/batch/submit-job.py \
    --region us-east-1 \
    --source-ref batch \
    --job-type g4dn.4x \
    --name convert_${MODEL_NAME} \
    --work-dir scripts/conversion_toolkits \
    --remote https://github.com/zheyuye/gluon-nlp/ \
    --command 'bash convert_'${MODEL_NAME}'.sh | tee stdout.log'
