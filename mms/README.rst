Multi-model-server example
==========================

https://github.com/awslabs/multi-model-server/

Assuming you are located in the root of the GluonNLP repo, you can run this
example via:

```
pip install --user multi-model-server
curl https://dist-bert.s3.amazonaws.com/demo/finetune/sst.params -o mms/sst.params
~/.local/bin/model-archiver --model-name bert_sst --model-path mms --handler bert:handle --runtime python --export-path /tmp
~/.local/bin/multi-model-server --start --models bert_sst.mar --model-store /tmp
curl -X POST http://127.0.0.1:8080/bert_sst/predict -F 'data=["Positive sentiment", "Negative sentiment"]'
```


