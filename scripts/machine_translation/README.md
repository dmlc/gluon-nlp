# Machine Translation

To run the scripts, you are recommended to install [tensorboardX](https://github.com/lanpa/tensorboardX).

```
python3 -m pip install tensorboardX
```

Also, to install horovod, you can try out the following command:

```
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_MXNET=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_NCCL_HOME=/opt/nccl/build CUDA_HOME=/usr/local/cuda pip install --no-cache-dir horovod
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

### Transformer Base

For "transformer_base" configuration, we provide the script that uses horovod + amp for training.

```bash
SUBWORD_ALGO=yttm
SRC=en
TGT=de
lr=0.0016
num_accumulated=8
max_num_tokens=4096
epochs=60
SAVE_DIR=transformer_base_wmt2014_en_de_${SUBWORD_ALGO}_${lr}_${num_accumulated}_${max_num_tokens}_${epochs}
horovodrun -np 4 -H localhost:4 python3  train_transformer.py \
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
    --optimizer_params "{\"beta1\": 0.9, \"beta2\": 0.98}" \
    --cfg transformer_base \
    --lr ${lr} \
    --num_accumulated ${num_accumulated} \
    --sampler BoundedBudgetSampler \
    --max_num_tokens ${max_num_tokens} \
    --epochs ${epochs} \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 123 \
    --fp16
```

After we have trained the model, we can use the average_checkpoint cli to average the last 10 checkpoints

```bash
gluon_average_checkpoint --checkpoints ${SAVE_DIR}/epoch*.params \
    --begin 21 \
    --end 30 \
    --save-path ${SAVE_DIR}/avg_21_30.params
```

Use the following command to inference/evaluate the Transformer model:

```bash
SUBWORD_ALGO=yttm
python3 evaluate_transformer.py \
    --param_path ${SAVE_DIR}/avg_25_29.params \
    --src_lang en \
    --tgt_lang de \
    --cfg ${SAVE_DIR}/config.yml \
    --src_tokenizer ${SUBWORD_ALGO} \
    --tgt_tokenizer ${SUBWORD_ALGO} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --src_corpus wmt2014_ende/test.raw.en \
    --tgt_corpus wmt2014_ende/test.raw.de
```

### Transformer Big

For "transformer_wmt_en_de_big" configuration

```bash
SUBWORD_ALGO=yttm
SRC=en
TGT=de
lr=0.0005
num_accumulated=16
max_num_tokens=3584
epochs=60
SAVE_DIR=transformer_big_wmt2014_en_de_${SUBWORD_ALGO}_${lr}_${max_num_tokens}_${num_accumulated}_${epochs}_20201216
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
gluon_average_checkpoint --checkpoints ${SAVE_DIR}/epoch*.params \
    --begin 21 \
    --end 30 \
    --save-path ${SAVE_DIR}/avg_21_30.params
```


Use the following command to inference/evaluate the Transformer model:

```bash
python3 evaluate_transformer.py \
    --param_path ${SAVE_DIR}/avg_21_30.params \
    --src_lang en \
    --tgt_lang de \
    --cfg ${SAVE_DIR}/config.yml \
    --src_tokenizer ${SUBWORD_ALGO} \
    --tgt_tokenizer ${SUBWORD_ALGO} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --src_corpus wmt2014_ende/test.raw.en \
    --tgt_corpus wmt2014_ende/test.raw.de
```


Test BLEU score (evaluated via SacreBLEU):

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


### Train with customized configuration

For example, pre-layer normalization (Pre-LN) has been shown to be more stable than the post layer-normalization. 
(See also ["On Layer Normalization in the Transformer Architecture"](https://proceedings.icml.cc/static/paper_files/icml/2020/328-Paper.pdf)). 
Post-LN has been the default architecture used in `transformer-base` and `transformer-large`. In addition, it has been shown that we can use a deep encoder and
a shallow decoder to improve the performance as in
["Deep Encoder, Shallow Decoder:Reevaluating the Speed-Quality Tradeoff in Machine Translation"](https://arxiv.org/pdf/2006.10369.pdf)
To train with Pre-LN + Deep-Shallow architecture, you can specify the [transformer_base_pre_ln_enc12_dec1.yml](transformer_base_pre_ln_enc12_dec1.yml) and train with the configuration.

```
SUBWORD_ALGO=yttm
SRC=en
TGT=de
lr=5e-4
wd=0.01
num_accumulated=4
max_num_tokens=4096
epochs=60
SAVE_DIR=transformer_base_ende_prenorm_enc12_dec1_${SUBWORD_ALGO}_${lr}_${wd}_${num_accumulated}_${max_num_tokens}_${epochs}
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
    --optimizer adamw \
    --optimizer_params "{\"beta1\": 0.9, \"beta2\": 0.98, \"epsilon\": 1e-6}" \
    --cfg transformer_base_pre_ln_enc12_dec1.yml \
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
