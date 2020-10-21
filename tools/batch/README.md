# Launch AWS Jobs
For contributors of GluonNLP, you can try to launch jobs via AWS Batch.
Once you've correctly configured the AWS CLI, you may use the following command in which `remote` and `source-ref` denote the repository url and branch name respectively.

```
python3 submit-job.py \
--region us-east-1 \
--job-type p3.2x \
--source-ref master \
--work-dir tools/batch \
--remote https://github.com/dmlc/gluon-nlp \
--command "python3 hello_world.py" \
--wait
```

# Updating the Docker for AWS Batch.

You may refer to the instruction in [GluonNLP Docker Support](../docker/README.md#ci-maintainer) for more information.

## Conversion Toolkits
Following the instruction of [converting scripts](../../scripts/conversion_toolkits), 
several pre-trained models could be converted through the corresponding conversion tool as below command where `${MODEL_TYPE}` could be selected from `[albert, bert, electra, mobilebert, bart, robert, xmlr]`.
```bash
bash run_batch_conversion ${MODEL_TYPE}
```
