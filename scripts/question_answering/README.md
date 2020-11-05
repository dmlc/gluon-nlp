# Question Answering Examples

# SQuAD
The finetuning scripts for [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) are available,
supporting a variety of pre-training models including [BERT](https://github.com/google-research/electra), [ALBERT](https://github.com/google-research/albert),
and [ELECTRA](https://github.com/google-research/bert). Free to choose one of them as `model_name`, listing below.

|               BERT               |          ALBERT          |        ELECTRA       |
|:--------------------------------:|:------------------------:|:--------------------:|
| google_en_cased_bert_base        | google_albert_base_v2    | google_electra_small |
| google_en_uncased_bert_base      | google_albert_large_v2   | google_electra_base  |
| google_en_cased_bert_large       | google_albert_xalrge_v2  | google_electra_large |
| google_en_uncased_bert_large     | google_albert_xxlarge_v2 |                      |
| google_en_cased_bert_wwm_large   |                          |                      |
| google_en_uncased_bert_wwm_large |                          |                      |

### Data and official evaluation scripts

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
*   [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
*   [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

download and move them under `$DATA_DIR`

### Running Script
We provide the script to train on the SQuAD dataset.

```bash
VERSION=2.0  # Either 2.0 or 1.1
MODEL_NAME=google_albert_base_v2

# Prepare the Data
nlp_data prepare_squad --version ${VERSION}

# Run the script
python3 run_squad.py \
    --model_name ${MODEL_NAME} \
    --data_dir squad \
    --output_dir fintune_${MODEL_NAME}_squad_${VERSION} \
    --version ${VERSION} \
    --do_eval \
    --do_train \
    --batch_size 4 \
    --num_accumulated 3 \
    --gpus 0,1,2,3 \
    --epochs 3 \
    --lr 2e-5 \
    --warmup_ratio 0.1 \
    --wd 0.01 \
    --max_seq_length 512 \
    --max_grad_norm 0.1 \
    --overwrite_cache \
```

To evaluate based on a fine-tuned checkpoint, we can use the following command:

```bash
CKPT_PATH=fintune_${MODEL_NAME}_squad_${VERSION}/google_albert_base_v2_squad2.0_8164.params
OUT_DIR=fintune_${MODEL_NAME}_squad_${VERSION}/evaluate
python3 run_squad.py \
    --model_name ${MODEL_NAME} \
    --data_dir squad \
    --output_dir ${OUT_DIR} \
    --param_checkpoint ${CKPT_PATH} \
    --version ${VERSION} \
    --do_eval \
    --gpus 0 \
    --eval_batch_size 16 \
    --overwrite_cache \
```

### Using Horovod

We could speed up multi-GPU training via [Horovod](https://github.com/horovod/horovod).
Compared to KVStore, training RoBERTa Large model on SQuAD 2.0 with 3 epochs will save 
roughly 1/4 training resources (8.48 vs 11.32 hours). Results may vary depending on the 
training instances.

```bash
horovodrun -np 4 -H localhost:4 python3 run_squad.py \
    --comm_backend horovod \
    ...
```

### Finetuning Details
As for ELECTRA model, we fine-tune it with layer-wise learning rate decay as

```bash
VERSION=2.0  # Either 2.0 or 1.1
MODEL_NAME=google_electra_small

python3 run_squad.py \
    --model_name ${MODEL_NAME} \
    --data_dir squad \
    --output_dir fintune_${MODEL_NAME}_squad_${VERSION} \
    --version ${VERSION} \
    --do_eval \
    --do_train \
    --batch_size 32 \
    --num_accumulated 1 \
    --gpus 0 \
    --epochs 2 \
    --lr 3e-4 \
    --layerwise_decay 0.8 \
    --warmup_ratio 0.1 \
    --wd 0 \
    --max_seq_length 512 \
    --max_grad_norm 0.1 \
```

For RoBERTa and XLMR, we remove 'segment_ids' and replace `[CLS]` and `[SEP]` with
`<s>` and `</s>` which stand for the beginning and end of sentences respectively in original purpose.

```bash
VERSION=2.0  # Either 2.0 or 1.1
MODEL_NAME=fairseq_roberta_large

python3 run_squad.py \
    --model_name ${MODEL_NAME} \
    --data_dir squad \
    --output_dir fintune_${MODEL_NAME}_squad_${VERSION} \
    --version ${VERSION} \
    --do_eval \
    --do_train \
    --batch_size 2 \
    --num_accumulated 6 \
    --gpus 0,1,2,3 \
    --epochs 3 \
    --lr 3e-5 \
    --warmup_ratio 0.2 \
    --wd 0.01 \
    --max_seq_length 512 \
    --max_grad_norm 0.1 \
```

### Results
We reproduced the ALBERT model which is released by Google, and fine-tune on SQuAD with a single model. 
ALBERT Version 2 are pre-trained without the dropout mechanism but with extra training steps compared to the version 1 (see the [original paper](https://arxiv.org/abs/1909.11942) for details).

Fine-tuning the listed models with hyper-parameter learning rate 2e-5, epochs 3, warmup ratio 0.1 and max gradient norm 0.1 (as shown in command). Notice that the `batch_size` is set for each GPU and the global batch size is 48 for all experiments, besides that gradient accumulation (`num_accumulated`) is supported in the case of out of memory.

Performance are shown in the table below, in which the SQuAD1.1 are evaluated with SQuAD2.0 checkpoints.
Notice that the standard metrics of SQuAD are `EM/F1`. The former is an exact match score between predictions and references, 
while the latter is a token-level F1 score in which the common tokens are considered as True Positives.

|Reproduced ALBERT Models (F1/EM)  | SQuAD 1.1 dev | SQuAD 2.0 dev | SQuAD 2.0 Results File | Log | Command | Weight |
|----------------------------------|---------------|---------------|------|-----|---------|----------|
|ALBERT base                       | 90.55/83.83   | 82.57/79.75   |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_base/fintune_google_albert_base_v2_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_base/fintune_google_albert_base_v2_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_albert_base.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_base/fintune_google_albert_base_v2_squad_2.0/google_albert_base_v2_squad2.0_8163.params) |
|ALBERT large                      | 92.66/86.43   | 85.21/82.50   |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_large/fintune_google_albert_large_v2_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_large/fintune_google_albert_large_v2_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_albert_large.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_large/fintune_google_albert_large_v2_squad_2.0/google_albert_large_v2_squad2.0_8163.params) |
|ALBERT xlarge                     | 93.85/87.71   | 87.73/84.83   |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_xlarge/fintune_google_albert_xlarge_v2_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_xlarge/fintune_google_albert_xlarge_v2_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_albert_xlarge.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_xlarge/fintune_google_albert_xlarge_v2_squad_2.0/google_albert_xlarge_v2_squad2.0_8163.params) |
|ALBERT xxlarge                    | 95.00/89.01   | 89.84/86.79   |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_xxlarge/fintune_google_albert_xxlarge_v2_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_xxlarge/fintune_google_albert_xxlarge_v2_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_albert_xxlarge.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_albert_xxlarge/fintune_google_albert_xxlarge_v2_squad_2.0/google_albert_xxlarge_v2_squad2.0_8163.params) |

For reference, we've included the results from Google's Original Experiments

| Model Name (F1/EM) | SQuAD 1.1 dev | SQuAD 2.0 dev|
|------------|---------------|--------------|
|ALBERT base (googleresearch/albert)    | 90.2/83.2     | 82.1/79.3    |
|ALBERT large (googleresearch/albert)   | 91.8/85.2     | 84.9/81.8    |
|ALBERT xlarge (googleresearch/albert)  | 92.9/86.4     | 87.9/84.1    |
|ALBERT xxlarge (googleresearch/albert) | 94.6/89.1     | 89.8/86.9    |

For the reset pretrained models, the results on SQuAD1.1 and SQuAD2.0 are given as follows.

| Model Name (F1/EM)    | SQuAD1.1 dev  | SQuAD2.0 dev | SQuAD 2.0 Results File | Log | Command | Weight |
|--------------------------|---------------|--------------|------|-----|--------|---------|
|BERT base                 | 88.44/81.54   | 76.32/73.64  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_uncased_bert_base/fintune_google_en_uncased_bert_base_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_uncased_bert_base/fintune_google_en_uncased_bert_base_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_uncased_bert_base.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_uncased_bert_base/fintune_google_en_uncased_bert_base_squad_2.0/google_en_uncased_bert_base_squad2.0_8160.params) |
|BERT large                | 90.65/84.02   | 81.22/78.22  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_uncased_bert_large/fintune_google_en_uncased_bert_large_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_uncased_bert_large/fintune_google_en_uncased_bert_large_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_uncased_bert_large.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_uncased_bert_large/fintune_google_en_uncased_bert_large_squad_2.0/google_en_uncased_bert_large_squad2.0_8159.params) |
|ELECTRA small             | 85.76/79.16   | 74.07/71.56  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_electra_small/fintune_google_electra_small_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_electra_small/fintune_google_electra_small_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_electra_small.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_electra_small/fintune_google_electra_small_squad_2.0/google_electra_small_squad2.0_8160.params) |
|ELECTRA base              | 92.64/86.99   | 86.33/83.67  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_electra_base/fintune_google_electra_base_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_electra_base/fintune_google_electra_base_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_electra_base.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_electra_base/fintune_google_electra_base_squad_2.0/google_electra_base_squad2.0_8160.params) |
|ELECTRA large             | 94.79/89.52   | 90.55/88.24  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_electra_large/fintune_google_electra_large_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_electra_large/fintune_google_electra_large_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_electra_large.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_electra_large/fintune_google_electra_large_squad_2.0/google_electra_large_squad2.0_8159.params) |
|MobileBERT                | 89.69/82.88   | 80.27/77.60  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_mobilebert/fintune_google_uncased_mobilebert_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_mobilebert/fintune_google_uncased_mobilebert_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_mobilebert.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_mobilebert/fintune_google_uncased_mobilebert_squad_2.0/google_uncased_mobilebert_squad2.0_20615.params) |
|RoBERTa large             | 94.57/88.88   | 89.70/86.79  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_roberta_large/fintune_fairseq_roberta_large_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_roberta_large/fintune_fairseq_roberta_large_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_electra_large.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201025/test_squad2_roberta_large/fintune_fairseq_roberta_large_squad_2.0/fairseq_roberta_large_squad2.0_8160.params) |

For reference, we have also included the results of original version from Google and Fairseq

| Model Name (F1/EM)       | SQuAD1.1 dev   | SQuAD2.0 dev  |
|--------------------------|----------------|---------------|
|Google BERT base          |   88.5/80.8    |     - / -     |
|Google BERT large         |   90.9/84.1    |     - / -     |
|Google ELECTRA small      |     - /75.8    |      -/70.1   |
|Google ELECTRA base       |      -/86.8    |      -/83.7   |
|Google ELECTRA large      |      -/89.7     |     -/88.1   |
|Google MobileBERT         |   90.0/82.9	|   79.2/76.2   |
|Fairseq RoBERTa large     |   94.6/88.9    |	89.4/86.5   |

### Run with AWS Batch
We can quickly run the squad finetuning via the [AWS Batch support](../../tools/batch).

The code is given in [run_batch_squad.sh](run_batch_squad.sh)

```bash
# AWS Batch training without horovod on SQuAD 2.0
bash batch/run_batch_squad.sh 0 2.0 submit_squad_v2_fp32.log float32

# AWS Batch training with horovod on SQuAD 2.0
bash batch/run_batch_squad.sh 1 2.0 submit_squad_v2_horovod_fp32.log float32

# AWS Batch training with horovod on SQuAD 1.1
bash batch/run_batch_squad.sh 1 1.1 submit_squad_v1_horovod_fp32.log float32
```

```bash
# AWS Batch training with horovod on SQuAD 2.0 + FP16
bash batch/run_batch_squad.sh 1 2.0 submit_squad_v2_horovod_fp16.log float16

# AWS Batch training with horovod on SQuAD 1.1 + FP16
bash batch/run_batch_squad.sh 1 1.1 submit_squad_v1_horovod_fp16.log float16
```

Also, after you have submitted the jobs, you may sync the results via
```bash
bash batch/sync_batch_result.sh submit_squad_v2.log squad_v2_no_horovod
bash batch/sync_batch_result.sh submit_squad_v2_horovod.log squad_v2_horovod
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
