# Question Answering Examples

# SQuAD
The finetuning scripts for [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) are available, supporting a variety of pre-training models including [BERT](https://github.com/google-research/electra), [ALBERT]((https://github.com/google-research/albert), and [ELECTRA](https://github.com/google-research/bert). Free to choose one of them as `model_name`, listing below.

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
python run_squad.py \
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
    --wd=0.01 \
    --max_seq_length 512 \
    --max_grad_norm 0.1 \
    --overwrite_cache \
```
or evaluate SQuAD1.1 based on a SQuAD2.0 fine-tuned checkpoint as

```bash
python run_squad.py \
    --model_name ${MODEL_NAME} \
    --data_dir=squad \
    --output_dir=${OUT_DIR} \
    --param_checkpoint=${CKPT_PATH} \
    --version 2.0 \
    --do_eval \
    --gpus 0,1,2,3,4,5,6,7 \
    --eval_batch_size 16 \
    --overwrite_cache \
```

### Results
We reproduced the ALBERT model which is released by Google, and fine-tune the the SQuAD with single models. ALBERT Version 2 are pre-trained without the dropout mechanism but with extra training steps compared to the version 1 (see the [original paper](https://arxiv.org/abs/1909.11942) for details).

Fine-tuning the listed models with hyper-parameter learning rate 2e-5, epochs 3, warmup ratio 0.1 and max gradient norm 0.1 (as shown in command). Notice that the `batch_size` is set for each GPU and the global batch size is 48 for all experiments, besides that gradient accumulation (`num_accumulated`) is supported in the case of out of memory.

Performance are shown in the table below, in which the SQuAD1.1 are evaluated with SQuAD2.0 checkpoints.

|Reproduced ALBERT Models  | SQuAD 1.1 dev  | SQuAD 2.0 dev |
|--------------------------|---------------|--------------|
|ALBERT base               | 90.55/83.83   | 82.13/79.19  |
|ALBERT large              | 92.66/86.43   | 84.93/82.02  |
|ALBERT xlarge             | 93.85/87.71   | 87.61/84.59  |
|ALBERT xxlarge            | 95.00/89.01   | 90.0/86.9    |

For reference, we've included the results from Google's Original Experiments

| Model Name | SQuAD 1.1 dev | SQuAD 2.0 dev|
|------------|---------------|--------------|
|ALBERT base (googleresearch/albert)    | 90.2/83.2     | 82.1/79.3    |
|ALBERT large (googleresearch/albert)   | 91.8/85.2     | 84.9/81.8    |
|ALBERT xlarge (googleresearch/albert)  | 92.9/86.4     | 87.9/84.1    |
|ALBERT xxlarge (googleresearch/albert) | 94.6/89.1     | 89.8/86.9    |

For BERT and ELECTRA model, the SQuAD 1.1 is evaluated with SQuAD 2.0 checkpoints, as the following results showcased.

| Model Name    | SQuAD1.1 dev  | SQuAD2.0 dev |
|--------------------------|---------------|--------------|
|BERT base                 | 88.40/81.24   | 76.89/74.01  |
|BERT large                | 90.45/83.55   | 81.89/78.77  |
|ELECTRA small             | 84.40/74.41   | 71.73/68.78  |        
|ELECTRA base              | 92.19/86.07   | 83.89/81.16  |
|ELECTRA large             | 94.35/88.50   | 89.68/87.05  |

For reference, we have also included the results of Google's original version

| Model Name               | SQuAD1.1 dev   | SQuAD2.0 dev  |
|Google BERT base          |   88.5/80.8    |     - / -     |
|Google BERT large         |   90.9/84.1    |     - / -     |
|Google ELECTRA base       |     - /75.8    |     - /70.1   |
|Google ELECTRA base       |     - /86.8    |     - /80.5   |
|Google ELECTRA large      |     - /89.7    |     - /88.1   |

All experiments done on AWS P3.16xlarge (8 x NVIDIA Tesla V100 16 GB)
