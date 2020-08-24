MODEL_NAME=$1

python3 submit-job.py \
    --region us-east-1 \
    --source-ref master \
    --job-type g4dn.12x \
    --name test_squad_${MODEL_NAME} \
    --work-dir scripts/question_answering \
    --remote https://github.com/dmlc/gluon-nlp/ \
    --command 'bash commands/run_squad2_'${MODEL_NAME}'.sh | tee stdout.log'
