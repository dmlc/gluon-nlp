# Machine Translation

## Train a Transformer from scratch
First, use the script described in [datasets/machine_translation](../datasets/machine_translation) 
to generate the dataset. Then, run `train_transformer.py` to train the model. 

In the following, we give the training script for WMT2014 EN-DE task with yttm tokenizer. 
You may first run the following command in [datasets/machine_translation](../datasets/machine_translation).
```bash
bash ../datasets/machine_translation/wmt2014_ende.sh yttm
```

Then, you can run the experiment.
For "transformer_base" configuration

```bash
SUBWORD_ALGO=yttm
SRC=en
TGT=de
python3 train_transformer.py \
    --train_src_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${SRC} \
    --train_tgt_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_src_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_tgt_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir transformer_base_wmt2014_en_de_${SUBWORD_ALGO} \
    --cfg transformer_base \
    --lr 0.002 \
    --sampler BoundedBudgetSampler \
    --max_num_tokens 2700 \
    --max_update 15000 \
    --save_interval_update 500 \
    --warmup_steps 6000 \
    --warmup_init_lr 0.0 \
    --seed 123 \
    --gpus 0,1,2,3
```

Or training via horovod
```
horovodrun -np 4 -H localhost:4 python3 train_transformer.py \
    --comm_backend horovod \
    --train_src_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${SRC} \
    --train_tgt_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_src_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_tgt_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir transformer_base_wmt2014_en_de_${SUBWORD_ALGO} \
    --cfg transformer_base \
    --lr 0.002 \
    --sampler BoundedBudgetSampler \
    --max_num_tokens 2700 \
    --max_update 15000 \
    --save_interval_update 500 \
    --warmup_steps 6000 \
    --warmup_init_lr 0.0 \
    --seed 123 \
    --gpus 0,1,2,3
```

Use the average_checkpoint cli to average the last 10 checkpoints

```bash
gluon_average_checkpoint --checkpoints transformer_base_wmt2014_en_de_${SUBWORD_ALGO}/epoch*.params \
    --begin 30 \
    --end 39 \
    --save-path transformer_base_wmt2014_en_de_${SUBWORD_ALGO}/epoch_avg_30_39.params
```

Use the following command to inference/evaluate the Transformer model:

```bash
python3 evaluate_transformer.py \
    --param_path transformer_base_wmt2014_en_de_${SUBWORD_ALGO}/epoch_avg_30_39.params \
    --src_lang en \
    --tgt_lang de \
    --cfg transformer_base_wmt2014_en_de_${SUBWORD_ALGO}/config.yml \
    --src_tokenizer ${SUBWORD_ALGO} \
    --tgt_tokenizer ${SUBWORD_ALGO} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --src_corpus wmt2014_ende/test.raw.en \
    --tgt_corpus wmt2014_ende/test.raw.de
```



For "transformer_wmt_en_de_big" configuration

```bash
SUBWORD_ALGO=yttm
SRC=en
TGT=de
python3 train_transformer.py \
    --train_src_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${SRC} \
    --train_tgt_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_src_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_tgt_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir transformer_big_wmt2014_en_de_${SUBWORD_ALGO} \
    --cfg transformer_wmt_en_de_big \
    --lr 0.001 \
    --sampler BoundedBudgetSampler \
    --max_num_tokens 3584 \
    --max_update 15000 \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 123 \
    --gpus 0,1,2,3
```

Use the average_checkpoint cli to average the last 10 checkpoints

```bash
gluon_average_checkpoint --checkpoints transformer_big_wmt2014_en_de_${SUBWORD_ALGO}/update*.params \
    --begin 21 \
    --end 30 \
    --save-path transformer_big_wmt2014_en_de_${SUBWORD_ALGO}/avg_21_30.params
```


Use the following command to inference/evaluate the Transformer model:

```bash
python3 evaluate_transformer.py \
    --param_path transformer_big_wmt2014_en_de_${SUBWORD_ALGO}/average_21_30.params \
    --src_lang en \
    --tgt_lang de \
    --cfg transformer_big_wmt2014_en_de_${SUBWORD_ALGO}/config.yml \
    --src_tokenizer ${SUBWORD_ALGO} \
    --tgt_tokenizer ${SUBWORD_ALGO} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --src_corpus wmt2014_ende/test.raw.en \
    --tgt_corpus wmt2014_ende/test.raw.de
```


Test BLEU score with 3 seeds (evaluated via sacre BLEU):

- transformer_base

(test bleu / valid bleu)
| Subword Model | #Params    | Seed = 123  | Seed = 1234 | Seed = 12345 |  Mean±std   |
|---------------|------------|-------------|-------------|--------------|-------------|
| yttm          |            | 26.50/26.29 | -           |  -           |  -          |
| hf_bpe        |            |  -          | -           |  -           |  -          |
| spm           |            |  -          | -           |  -           |  -          |

- transformer_wmt_en_de_big

(test bleu / valid bleu)
| Subword Model | #Params    | Seed = 123  | Seed = 1234 | Seed = 12345 |  Mean±std   |
|---------------|------------|-------------|-------------|--------------|-------------|
| yttm          |            | 27.93/26.82 | -           |  -           |  -          |
| hf_bpe        |            |  -          | -           |  -           |  -          |
| spm           |            |  -          | -           |  -           |  -          |
