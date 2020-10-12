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

Our current batch job dockers are in 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1. To
update the docker:
- Update the Dockerfile
- Make sure docker and docker-compose, as well as the docker python package are installed.
- Export the AWS account credentials as environment variables
- CD to the same folder as the Dockerfile and execute the following:

```
# this executes a command that logs into ECR.
$(aws ecr get-login --no-include-email --region us-east-1)

# builds the docker image use the command in docker

# tags the recent build as gluon-nlp-1:latest, which AWS batch pulls from.
docker tag gluonai/gluon-nlp:gpu-ci-latest 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:gpu-ci-latest
docker tag gluonai/gluon-nlp:cpu-ci-latest 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:cpu-ci-latest

# pushes the change
docker push 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:gpu-ci-latest
docker push 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:cpu-ci-latest
```

## Conversion Toolkits
Following the instruction of [converting scripts](../../scripts/conversion_toolkits), 
several pre-trained models could be converted through the corresponding conversion tool as below command where `${MODEL_TYPE}` could be selected from `[albert, bert, electra, mobilebert, bart, robert, xmlr]`.
```bash
bash run_batch_conversion ${MODEL_TYPE}
```

## Fine-tuning Downstream Tasks

### Question Answering
We can quickly run the squad finetuning via [squad fine-tuning scripts](../../scripts/question_answering#squad) and the AWS Batch job.

The code is given in [run_batch_squad.sh](run_batch_squad.sh)

```bash
# AWS Batch training without horovod on SQuAD 2.0
bash run_batch_squad.sh

# AWS Batch training with horovod on SQuAD 2.0
bash run_batch_squad.sh 1 2.0 submit_squad_v2_horovod.log
```



Internally, it will train the following models on SQuAD 2.0 dataset:
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
| roberta_large      |
| mobilebert         |
