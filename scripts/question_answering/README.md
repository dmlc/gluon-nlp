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
| google_zh_bert_base              |                          |                      |
| google_multi_cased_bert_base     |                          |                      |
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
or evaluate SQuAD1.1 based on a SQuAD2.0 fine-tuned checkpoint as

```bash
python3 run_squad.py \
    --model_name ${MODEL_NAME} \
    --data_dir squad \
    --output_dir ${OUT_DIR} \
    --param_checkpoint ${CKPT_PATH} \
    --version 2.0 \
    --do_eval \
    --gpus 0,1,2,3 \
    --eval_batch_size 16 \
    --overwrite_cache \
```

We could speed up multi-GPU training via horovod.
Compared to KVStore, training RoBERTa Large model on SQuAD 2.0 with 3 epochs will save roughly 1/4 training resources (8.48 vs 11.32 hours). Results may vary depending on the training instances.

```bash
mpirun -np 4 -H localhost:4 python3 run_squad.py \
    --comm_backend horovod \
    ...
```
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
We reproduced the ALBERT model which is released by Google, and fine-tune the the SQuAD with single models. ALBERT Version 2 are pre-trained without the dropout mechanism but with extra training steps compared to the version 1 (see the [original paper](https://arxiv.org/abs/1909.11942) for details).

Fine-tuning the listed models with hyper-parameter learning rate 2e-5, epochs 3, warmup ratio 0.1 and max gradient norm 0.1 (as shown in command). Notice that the `batch_size` is set for each GPU and the global batch size is 48 for all experiments, besides that gradient accumulation (`num_accumulated`) is supported in the case of out of memory.

Performance are shown in the table below, in which the SQuAD1.1 are evaluated with SQuAD2.0 checkpoints.
Notice that the standard metrics of SQuAD are EM and F1. The former is an exact match score between predictions and references, while the latter is a token-level f1 score in which the common tokens are considered as True Positives.

|Reproduced ALBERT Models (F1/EM)  | SQuAD 1.1 dev | SQuAD 2.0 dev |
|----------------------------------|---------------|---------------|
|ALBERT base                       | 90.55/83.83   | 82.13/79.19   |
|ALBERT large                      | 92.66/86.43   | 84.93/82.02   |
|ALBERT xlarge                     | 93.85/87.71   | 87.61/84.59   |
|ALBERT xxlarge                    | 95.00/89.01   | 90.0/86.9     |

For reference, we've included the results from Google's Original Experiments

| Model Name | SQuAD 1.1 dev | SQuAD 2.0 dev|
|------------|---------------|--------------|
|ALBERT base (googleresearch/albert)    | 90.2/83.2     | 82.1/79.3    |
|ALBERT large (googleresearch/albert)   | 91.8/85.2     | 84.9/81.8    |
|ALBERT xlarge (googleresearch/albert)  | 92.9/86.4     | 87.9/84.1    |
|ALBERT xxlarge (googleresearch/albert) | 94.6/89.1     | 89.8/86.9    |

For the reset pretrained models, the results on SQuAD1.1 and SQuAD2.0 are given as follows.

| Model Name    | SQuAD1.1 dev  | SQuAD2.0 dev |
|--------------------------|---------------|--------------|
|BERT base                 | 88.40/81.24   | 76.89/74.01  |
|BERT large                | 90.45/83.55   | 81.89/78.77  |
|ELECTRA small             | 85.42/78.95   | 74.44/71.86  |        
|ELECTRA base              | 92.63/87.34   | 86.34/83.62  |
|ELECTRA large             | 94.95/89.94   | 90.59/88.13  |
|RoBERTa large             | 94.58/88.86   | 89.01/85.93  |

For reference, we have also included the results of original version from Google and Fairseq

| Model Name               | SQuAD1.1 dev   | SQuAD2.0 dev  |
|--------------------------|----------------|---------------|
|Google BERT base          |   88.5/80.8    |     - / -     |
|Google BERT large         |   90.9/84.1    |     - / -     |
|Google ELECTRA small      |     - /75.8    |     - /70.1   |
|Google ELECTRA base       |     - /86.8    |     - /83.7   |
|Google ELECTRA large      |     - /89.7    |     - /88.1   |
|Fairseq RoBERTa large     |   94.6/88.9    |	89.4/86.5   |
