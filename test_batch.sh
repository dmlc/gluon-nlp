python3 tools/batch/submit-job.py \
  --region us-east-1 \
  --job-type g4dn.4x \
  --name test_conversion \
  --work-dir scripts/conversion_toolkits/ \
  --command 'bash convert_bert_from_tf_hub.sh | tee stdout.log' \
  --wait 
