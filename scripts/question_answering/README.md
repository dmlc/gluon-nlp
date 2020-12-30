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

In addition, to train models with pre-chosen hyper-parameters, you can try out the scripts in [commands](./commands).

```
# Run FP32 training on SQuAD 2.0
bash commands/run_squad2_albert_base.sh 0 2.0 float32

# Run HOROVOD + FP32 training on SQuAD 2.0
bash commands/run_squad2_albert_base.sh 1 2.0 float32

# Run HOROVOD + AMP on SQuAD 2.0
bash commands/run_squad2_albert_base.sh 1 2.0 float16
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

Performance are shown in the table below. Notice that the standard metrics of SQuAD are `EM/F1`. The former is an exact match score between predictions and references, 
while the latter is a token-level F1 score in which the common tokens are considered as True Positives.

|Reproduced ALBERT Models (F1/EM)  | SQuAD 1.1 dev | SQuAD 2.0 dev | SQuAD 2.0 Results | Log | Command | SQuAD 2.0 Weight | SQuAD 1.0 Weight |
|----------------------------------|---------------|---------------|-------------------|-------------------|-----|---------|-----------------|
|ALBERT base                       | 90.37/83.57   | 82.38/79.49   | [json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_base/fintune_google_albert_base_v2_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_base/fintune_google_albert_base_v2_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_albert_base.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_base/fintune_google_albert_base_v2_squad_2.0/google_albert_base_v2_squad2.0_8163.params) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v1_horovod_fp32/test_squad2_albert_base/fintune_google_albert_base_v2_squad_1.1/google_albert_base_v2_squad1.1_5486.params) |
|ALBERT large                      | 92.68/86.57   | 85.05/82.18   | [json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_large/fintune_google_albert_large_v2_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_large/fintune_google_albert_large_v2_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_albert_large.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_large/fintune_google_albert_large_v2_squad_2.0/google_albert_large_v2_squad2.0_8163.params) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v1_horovod_fp32/test_squad2_albert_large/fintune_google_albert_large_v2_squad_1.1/google_albert_large_v2_squad1.1_5485.params) |
|ALBERT xlarge                     | 93.71/87.49   | 87.78/84.79   | [json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_xlarge/fintune_google_albert_xlarge_v2_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_xlarge/fintune_google_albert_xlarge_v2_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_albert_xlarge.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_xlarge/fintune_google_albert_xlarge_v2_squad_2.0/google_albert_xlarge_v2_squad2.0_8163.params) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v1_horovod_fp32/test_squad2_albert_xlarge/fintune_google_albert_xlarge_v2_squad_1.1/google_albert_xlarge_v2_squad1.1_5485.params) |
|ALBERT xxlarge                    | 94.50/88.44   | 90.29/87.29   | [json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_xxlarge/fintune_google_albert_xxlarge_v2_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_xxlarge/fintune_google_albert_xxlarge_v2_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_albert_xxlarge.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_xxlarge/fintune_google_albert_xxlarge_v2_squad_2.0/google_albert_xxlarge_v2_squad2.0_8163.params) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v1_horovod_fp32/test_squad2_albert_xxlarge/fintune_google_albert_xxlarge_v2_squad_1.1/google_albert_xxlarge_v2_squad1.1_5485.params) |

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
|BERT base                 | 88.44/81.54   | 76.29/73.52  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_uncased_bert_base/fintune_google_en_uncased_bert_base_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_uncased_bert_base/fintune_google_en_uncased_bert_base_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_uncased_bert_base.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_uncased_bert_base/fintune_google_en_uncased_bert_base_squad_2.0/google_en_uncased_bert_base_squad2.0_8160.params) |
|BERT large                | 90.65/84.02   | 81.41/78.61  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_uncased_bert_large/fintune_google_en_uncased_bert_large_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_uncased_bert_large/fintune_google_en_uncased_bert_large_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_uncased_bert_large.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_uncased_bert_large/fintune_google_en_uncased_bert_large_squad_2.0/google_en_uncased_bert_large_squad2.0_8159.params) |
|ELECTRA small             | 85.76/79.16   | 73.96/71.36  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_small/fintune_google_electra_small_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_small/fintune_google_electra_small_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_electra_small.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_small/fintune_google_electra_small_squad_2.0/google_electra_small_squad2.0_8160.params) |
|ELECTRA base              | 92.64/86.99   | 86.49/83.80  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_base/fintune_google_electra_base_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_base/fintune_google_electra_base_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_electra_base.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_base/fintune_google_electra_base_squad_2.0/google_electra_base_squad2.0_8160.params) |
|ELECTRA large             | 94.79/89.52   | 90.61/88.18  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_large/fintune_google_electra_large_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_large/fintune_google_electra_large_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_electra_large.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_large/fintune_google_electra_large_squad_2.0/google_electra_large_squad2.0_8159.params) |
|MobileBERT                | 89.69/82.88   | 80.33/77.76  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_mobilebert/fintune_google_uncased_mobilebert_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_mobilebert/fintune_google_uncased_mobilebert_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_mobilebert.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_mobilebert/fintune_google_uncased_mobilebert_squad_2.0/google_uncased_mobilebert_squad2.0_20615.params) |
|RoBERTa large             | 94.57/88.88   | 89.35/86.46  |[json](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_roberta_large/fintune_fairseq_roberta_large_squad_2.0/best_results.json) | [log](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_roberta_large/fintune_fairseq_roberta_large_squad_2.0/finetune_squad2.0.log) | [command](./commands/run_squad2_electra_large.sh) | [weight](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_roberta_large/fintune_fairseq_roberta_large_squad_2.0/fairseq_roberta_large_squad2.0_8160.params) |

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

### Using AMP for faster training

Just add `--dtype float16` if you'd like to use AMP for training. Also, it will use 
half-precision for inference.

The following are the results obtained by combining AMP and horovod for training.


| Model Name (F1/EM)           | SQuAD1.1 dev | SQuAD2.0 dev | Weights |
|------------------------------|--------------|--------------|-------------------|
| ALBERT base                  | 90.48/83.86  | 82.40/79.50  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_base/fintune_google_albert_base_v2_squad_2.0/google_albert_base_v2_squad2.0_8163.params) | 
| ALBERT large                 | 92.55/86.27  | 85.10/82.19  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_large/fintune_google_albert_large_v2_squad_2.0/google_albert_large_v2_squad2.0_8163.params) |
| ALBERT xlarge                | 93.72/87.65  | 88.20/85.29  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_xlarge/fintune_google_albert_xlarge_v2_squad_2.0/google_albert_xlarge_v2_squad2.0_8163.params) |
| ALBERT xxlarge               | 94.68/88.69  | 90.13/87.19  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_albert_xxlarge/fintune_google_albert_xxlarge_v2_squad_2.0/google_albert_xxlarge_v2_squad2.0_8163.params) |
| ELECTRA base                 | 92.75/87.20  | 86.50/83.87  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_base/fintune_google_electra_base_squad_2.0/google_electra_base_squad2.0_8160.params) |
| ELECTRA large                | 94.92/89.81  | 90.54/88.07  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_large/fintune_google_electra_large_squad_2.0/google_electra_large_squad2.0_8159.params) |
| ELECTRA small                | 85.82/79.12  | 74.10/71.38  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_electra_small/fintune_google_electra_small_squad_2.0/google_electra_small_squad2.0_8160.params) |
| GluonNLP BERT Cased Base V1  | 88.60/81.92  | 78.15/75.22  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_gluon_en_cased_bert_base_v1/fintune_gluon_en_cased_bert_base_v1_squad_2.0/fintune_gluon_en_cased_bert_base_v1_squad_2.0_8163.params) |
| Fairseq RoBERTa large        | 94.59/88.80  | 89.44/86.50  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_roberta_large/fintune_google_fairseq_roberta_large_squad_2.0/fairseq_roberta_large_squad2.0_8161.params) |
| Google BERT base             | 88.31/81.22  | 76.45/73.60  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_uncased_bert_base/fintune_google_uncased_bert_base_squad_2.0/google_uncased_bert_base_squad2.0_8160.params) |
| Google BERT large            | 90.61/83.81  | 81.52/78.60  | [SQuAD2.0](https://gluon-nlp-log.s3.amazonaws.com/squad_training_log/20201229/squad_v2_horovod_fp32/test_squad2_uncased_bert_large/fintune_google_uncased_bert_large_squad_2.0/google_uncased_bert_large_squad2.0_8159.params) |
