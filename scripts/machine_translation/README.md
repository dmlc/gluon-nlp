# Machine Translation

To run the scripts, you are recommended to install [tensorboardX](https://github.com/lanpa/tensorboardX).

```
python3 -m pip install tensorboardX
```

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
lr=0.0016
num_accumulated=16
max_num_tokens=4096
epochs=60
SAVE_DIR=transformer_base_wmt2014_en_de_${SUBWORD_ALGO}_${lr}_${num_accumulated}_${max_num_tokens}_${epochs}
python3 train_transformer.py \
    --train_src_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${SRC} \
    --train_tgt_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_src_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_tgt_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir ${SAVE_DIR} \
    --cfg transformer_base \
    --lr ${lr} \
    --num_accumulated ${num_accumulated} \
    --sampler BoundedBudgetSampler \
    --max_num_tokens ${max_num_tokens} \
    --epochs ${epochs} \
    --warmup_steps 4000 \
    --warmup_init_lr 1e-07 \
    --seed 123 \
    --fp16 \
    --gpus 0,1,2,3
```

Or training via horovod by launching the job with 
```
horovodrun -np 4 -H localhost:4 python3 train_transformer.py --comm_backend horovod ...  

```

For example, with horovod + amp training, the previous command will become
```
SUBWORD_ALGO=yttm
SRC=en
TGT=de
lr=0.0016
num_accumulated=16
max_num_tokens=4096
epochs=60
SAVE_DIR=transformer_base_wmt2014_en_de_${SUBWORD_ALGO}_${lr}_${num_accumulated}_${max_num_tokens}_${epochs}
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
    --save_dir ${SAVE_DIR} \
    --cfg transformer_base \
    --lr ${lr} \
    --num_accumulated ${num_accumulated} \
    --sampler BoundedBudgetSampler \
    --max_num_tokens ${max_num_tokens} \
    --epochs ${epochs} \
    --warmup_steps 4000 \
    --warmup_init_lr 1e-07 \
    --seed 123 \
    --fp16
```

Use the average_checkpoint cli to average the last 5 checkpoints

```bash
gluon_average_checkpoint --checkpoints ${SAVE_DIR}/epoch*.params \
    --begin 50 \
    --end 60 \
    --save-path ${SAVE_DIR}/avg_50_60.params
```

Use the following command to inference/evaluate the Transformer model:

```bash
SUBWORD_ALGO=yttm
python3 evaluate_transformer.py \
    --param_path transformer_base_wmt2014_en_de_${SUBWORD_ALGO}/avg_25_29.params \
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
lr=0.0008
num_accumulated=16
max_num_tokens=4096
epochs=30
SAVE_DIR=transformer_big_wmt2014_en_de_${SUBWORD_ALGO}_${lr}_${max_num_tokens}_${num_accumulated}_${epochs}
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
    --save_dir ${SAVE_DIR} \
    --cfg transformer_wmt_en_de_big \
    --lr ${lr} \
    --num_accumulated ${num_accumulated} \
    --max_num_tokens ${max_num_tokens} \
    --sampler BoundedBudgetSampler \
    --epochs ${epochs} \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 123 \
    --fp16
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
    --param_path transformer_big_wmt2014_en_de_${SUBWORD_ALGO}/avg_21_30.params \
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
| Subword Model | Seed = 123  |
|---------------|-------------|
| yttm          | 26.95/25.85 |

- transformer_wmt_en_de_big

(test bleu / valid bleu)
| Subword Model | Seed = 123  |
|---------------|-------------|
| yttm          | 27.99/26.84 |


## Train with customized configuration

For example, pre-LayerNormalization (Pre-LN) has been shown to be more stable than the Post-LN 
(See also ["On Layer Normalization in the Transformer Architecture"](https://proceedings.icml.cc/static/paper_files/icml/2020/328-Paper.pdf)). 
Post-LN has been the default architecture used in `transformer-base` and `transformer-large`. 
To train with Pre-LN, you can specify the [transformer_base_pre_ln.yml](./transformer_base_pre_ln.yml) and train with the configuration.

```
SUBWORD_ALGO=yttm
SRC=en
TGT=de
lr=0.002
num_accumulated=16
max_num_tokens=4096
epochs=60
SAVE_DIR=transformer_base_ende_prenorm_${SUBWORD_ALGO}_${lr}_${num_accumulated}_${max_num_tokens}_${epochs}
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
    --save_dir ${SAVE_DIR} \
    --cfg transformer_base_pre_ln.yml \
    --lr ${lr} \
    --num_accumulated ${num_accumulated} \
    --sampler BoundedBudgetSampler \
    --max_num_tokens ${max_num_tokens} \
    --epochs ${epochs} \
    --warmup_steps 4000 \
    --warmup_init_lr 1e-07 \
    --seed 123 \
    --fp16
```

(test bleu)
| Subword Model | Seed = 123  |
|---------------|-------------|
| yttm          | 26.81 |
