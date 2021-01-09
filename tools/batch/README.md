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

## SQuAD Training

The code is given in [question_answering/run_batch_squad.sh](question_answering/run_batch_squad.sh)

```bash
# AWS Batch training without horovod on SQuAD 2.0
bash question_answering/run_batch_squad.sh 0 2.0 submit_squad_v2_fp32.log float32

# AWS Batch training with horovod on SQuAD 2.0
bash question_answering/run_batch_squad.sh 1 2.0 submit_squad_v2_horovod_fp32.log float32

# AWS Batch training with horovod on SQuAD 1.1
bash question_answering/run_batch_squad.sh 1 1.1 submit_squad_v1_horovod_fp32.log float32
```

```bash
# AWS Batch training with horovod on SQuAD 2.0 + FP16
bash question_answering/run_batch_squad.sh 1 2.0 submit_squad_v2_horovod_fp16.log float16

# AWS Batch training with horovod on SQuAD 1.1 + FP16
bash question_answering/run_batch_squad.sh 1 1.1 submit_squad_v1_horovod_fp16.log float16
```

Also, after you have submitted the jobs, you may sync the results via
```bash
bash sync_batch_result.sh submit_squad_v2_fp32.log squad_v2_no_horovod
bash sync_batch_result.sh submit_squad_v2_horovod_fp32.log squad_v2_horovod_fp32
bash sync_batch_result.sh submit_squad_v2_horovod_fp16.log squad_v2_horovod_fp16
```

You can then use [parse_squad_results.py](question_answering/parse_squad_results.py) to parse the 
results of different models to a single `.csv` file.

```bash
python3 question_answering/parse_squad_results.py --dir squad_v2_horovod_fp32
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
| gluon_en_cased_bert_base_v1    |
| mobilebert         |

## Benchmarking NLP Backbones
```bash
# Test for g4dn.4x
bash backbone_benchmark/run_batch_backbone_benchmark.sh g4dn.4x g4dn.4x_result.log

# Test for p3.2x
bash backbone_benchmark/run_batch_backbone_benchmark.sh p3.2x p3.2x_result.log
```

After these jobs are finished, you can download the benchmarking results via

```
bash sync_batch_result.sh g4dn.4x_result.log g4dn.4x_benchmark
bash sync_batch_result.sh p3.2x_result.log p3.2x_benchmark
```
