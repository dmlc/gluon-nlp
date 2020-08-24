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

## Conversion Toolkits
Following the instruction of [converting scripts](../../scripts/conversion_toolkits), several pre-trained models could be converted through the corresponding conversion tool as below command where `${MODEL_TYPE}` could be selected from `[albert, bert, electra, mobilebert, bart, robert, xmlr]`.
```bash
bash run_batch_conversion ${MODEL_TYPE}
```
## Fine-tuning Downstream Tasks

### Question Answering
We can quickly deploy an experiment via [squad fine-tuning scripts](../../scripts/question_answering#squad) as

```bash
bash run_batch_squad.sh ${MODEL_NAME}
```

in which `${MODEL_NAME}` is the name of available pre-trained models listing as following:
|    MODEL_NAME      |
|:------------------:|
| uncased_bert_base  |
| uncased_bert_large |
| albert_base        |
| albert_large       |
| albert_xlarge      |  
| albert_xxlarge     |
| electra_small      |
| electra_base       |
| electra_large      |
| roberta_base       |
| roberta_large      |
| mobilebert         |

### Machine Translation

### Text Translation

## Pre-trained Model Training
