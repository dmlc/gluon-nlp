for MODEL_NAME in albert_base albert_large albert_xlarge albert_xxlarge electra_base electra_large electra_small roberta_large uncased_bert_base uncased_bert_large
do
  python3 submit-job.py \
      --region us-east-1 \
      --source-ref fix_data_nmt_test \
      --job-type g4dn.12x \
      --name test_squad_${MODEL_NAME} \
      --work-dir scripts/question_answering \
      --remote https://github.com/sxjscience/gluon-nlp/ \
      --command 'bash commands/run_squad2_'${MODEL_NAME}'.sh | tee stdout.log' >> submit_squad_v2.log
done
