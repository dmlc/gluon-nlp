for MODEL_NAME in bert albert electra mobilebert roberta xlmr bart
do
  python3 submit-job.py \
      --region us-east-1 \
      --source-ref master \
      --job-type c5n.4x \
      --name convert_${MODEL_NAME} \
      --work-dir scripts/conversion_toolkits \
      --remote https://github.com/dmlc/gluon-nlp/ \
      --command 'bash convert_'${MODEL_NAME}'.sh | tee stdout.log' >> log.info
done
