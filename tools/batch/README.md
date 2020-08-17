# Launch AWS Jobs
For contributors of GluonNLP, you can try to launch jobs via AWS Batch. 
Once you've correctly configured the AWS CLI, you may use the following command:

```
python3 submit-job.py \
--region us-east-1 \
--job-type p3.2x \
--work-dir tools/batch \
--remote https://github.com/dmlc/gluon-nlp \
--command "python3 hello_world.py" \
--wait
```
