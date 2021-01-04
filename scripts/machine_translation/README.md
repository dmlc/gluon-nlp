# Machine Translation

To run the scripts, you are recommended to install [tensorboardX](https://github.com/lanpa/tensorboardX).

```
python3 -m pip install tensorboardX
```

Also, to install horovod, you can try out the following command:

```
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_MXNET=1 HOROVOD_WITHOUT_TENSORFLOW=1 pip install --no-cache-dir horovod
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
num_accumulated=16
max_num_tokens=4096
epochs=60
SAVE_DIR=transformer_base_wmt2014_en_de_${SUBWORD_ALGO}_${lr}_${num_accumulated}_${max_num_tokens}_${epochs}
horovodrun -np 4 -H localhost:4 python3  train_transformer.py \
    --comm_backend horovod \
    --train_src_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${SRC} \
    --train_tgt_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${TGT} \
    --src_lang ${SRC} \
    --tgt_lang ${TGT} \
    --src_tokenizer ${SUBWORD_ALGO} \
    --tgt_tokenizer ${SUBWORD_ALGO} \
    --dev_src_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_tgt_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_tgt_raw_corpus wmt2014_ende/dev.raw.${TGT} \
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
    --warmup_init_lr 1e-07 \
    --seed 123 \
    --max_grad_norm 1.0 \
    --fp16
```

After we have trained the model, we can use the average_checkpoint cli to average the last 10 checkpoints

```bash
gluon_average_checkpoint --checkpoints ${SAVE_DIR}/epoch*.params \
    --begin 51 \
    --end 60 \
    --save-path ${SAVE_DIR}/avg_51_60.params
```

Use the following command to inference/evaluate the Transformer model:

```bash
SUBWORD_ALGO=yttm
python3 evaluate_transformer.py \
    --param_path ${SAVE_DIR}/avg_51_60.params \
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
    --tgt_corpus wmt2014_ende/test.raw.de \
    --lp_alpha 0.0 \
    --fp16
```

In addition, to analyze the test/dev scores of models saved at different epochs, you can use

```bash
bash evaluate_epochs_wmt2014_ende.sh ${SAVE_DIR}
```

#### Results

We evaluate the results via SacreBLEU and attach the result + hash.
l

```
cat pred_sentences.txt | sacrebleu -t wmt14/full -l en-de
```

| Subword Model | Beam Search | Seed | Test BLEU | Tensorboard | Weights | Log | Config |
|---------------|-------------|------|-----------|-------------|---------|-----|--------|
| yttm          | lp_alpha=0.0, beam=4  | 123 | 27.06  | [tensorboard](https://tensorboard.dev/experiment/14F0u9V0SDGL2LSpNyo1hw/) | [weight](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_base_wmt2014_en_de_yttm_0.0016_16_4096_60_20201228/avg_51_60.params) | [log](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_base_wmt2014_en_de_yttm_0.0016_16_4096_60_20201228/train_transformer_rank0_local0_4.log) | [config](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_base_wmt2014_en_de_yttm_0.0016_16_4096_60_20201228/config.yml) |

The sacreBLEU hash is
```
BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.14 = 27.1 58.2/32.7/20.7/13.6 (BP = 1.000 ratio = 1.025 hyp_len = 64226 ref_len = 62688)
```


### Transformer Big

For "transformer_wmt_en_de_big" configuration

```bash
SUBWORD_ALGO=yttm
SRC=en
TGT=de
lr=0.0006
num_accumulated=16
max_num_tokens=3584
adam_epsilon=1e-9
epochs=60
SAVE_DIR=transformer_big_wmt2014_en_de_${SUBWORD_ALGO}_${lr}_${max_num_tokens}_${num_accumulated}_${epochs}_eps${adam_epsilon}_norm_clip
horovodrun -np 4 -H localhost:4 python3 train_transformer.py \
    --comm_backend horovod \
    --train_src_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${SRC} \
    --train_tgt_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${TGT} \
    --src_lang ${SRC} \
    --tgt_lang ${TGT} \
    --src_tokenizer ${SUBWORD_ALGO} \
    --tgt_tokenizer ${SUBWORD_ALGO} \
    --dev_src_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_tgt_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_tgt_raw_corpus wmt2014_ende/dev.raw.${TGT} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir ${SAVE_DIR} \
    --optimizer_params "{\"epsilon\": ${adam_epsilon}}" \
    --cfg transformer_wmt_en_de_big \
    --lr ${lr} \
    --num_accumulated ${num_accumulated} \
    --max_num_tokens ${max_num_tokens} \
    --sampler BoundedBudgetSampler \
    --epochs ${epochs} \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 123 \
    --max_grad_norm 1.0 \
    --fp16
```

Use the average_checkpoint cli to average the last 15 checkpoints. 

```bash
gluon_average_checkpoint --checkpoints ${SAVE_DIR}/epoch*.params \
    --begin 46 \
    --end 60 \
    --save-path ${SAVE_DIR}/avg_46_60.params
```


Use the following command to inference/evaluate the Transformer model. 

```bash
SUBWORD_ALGO=yttm
python3 evaluate_transformer.py \
    --param_path ${SAVE_DIR}/avg_46_60.params \
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
    --tgt_corpus wmt2014_ende/test.raw.de \
    --lp_alpha 0.0 \
    --fp16
```

#### Results

| Subword Model | Beam Search | Seed  | Test BLEU | Tensorboard | Weights | Log | Config |
|---------------|-------------|-------|-----------|-------------|---------|-----|--------|
| yttm          | lp_alpha 0.0, beam 4  | 123 | 28.01 | [tensorboard](https://tensorboard.dev/experiment/jm4bs3f3Q9CRYN3SxLBi9Q) | [weight](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_big_wmt2014_en_de_yttm_0.0006_3584_16_60_eps1e-9_norm_clip_20201227/avg_46_60.params) | [log](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_big_wmt2014_en_de_yttm_0.0006_3584_16_60_eps1e-9_norm_clip_20201227/train_transformer_rank0_local0_4.log) | [config](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_big_wmt2014_en_de_yttm_0.0006_3584_16_60_eps1e-9_norm_clip_20201227/config.yml) |

The sacreBLEU hash is
```
BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.14 = 28.0 58.9/33.7/21.6/14.4 (BP = 1.000 ratio = 1.023 hyp_len = 64110 ref_len = 62688)
```

### Other configurations

#### Pre-layer normalization

Pre-layer normalization (Pre-LN) has been shown to be more stable than the post layer-normalization. 
(See also ["On Layer Normalization in the Transformer Architecture"](http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf), 
and ["Understanding the Difficulty of Training Transformers"](https://www.aclweb.org/anthology/2020.emnlp-main.463.pdf)). 
Post-LN has been the default architecture used in `transformer-base` and `transformer-large`. 
To use the pre-norm variant, we can use the big-architecture implemented in Tensor2tensor: `transformer_wmt_en_de_big_t2t`.

```
SUBWORD_ALGO=yttm
SRC=en
TGT=de
lr=0.0006
num_accumulated=4
max_num_tokens=2560
adam_epsilon=1e-9
epochs=60
SAVE_DIR=transformer_big_t2t_wmt2014_${SRC}_${TGT}_${SUBWORD_ALGO}_${lr}_${max_num_tokens}_${num_accumulated}_${epochs}_eps${adam_epsilon}
horovodrun -np 4 -H localhost:4 python3 train_transformer.py \
    --comm_backend horovod \
    --train_src_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${SRC} \
    --train_tgt_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${TGT} \
    --src_lang ${SRC} \
    --tgt_lang ${TGT} \
    --src_tokenizer ${SUBWORD_ALGO} \
    --tgt_tokenizer ${SUBWORD_ALGO} \
    --dev_src_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_tgt_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_tgt_raw_corpus wmt2014_ende/dev.raw.${TGT} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir ${SAVE_DIR} \
    --optimizer_params "{\"epsilon\": ${adam_epsilon}}" \
    --cfg transformer_wmt_en_de_big_t2t \
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

Similarly, we will average the checkpoints and run evaluation (as described in [Transformer Big](#transformer-big))

We evaluated with SacreBLEU and also attached the hash

| Subword Model | Beam Search | Seed  | Test BLEU | Tensorboard | Weights | Log | Config |
|---------------|-------------|-------|-----------|-------------|---------|-----|--------|
| yttm          | lp_alpha 0.0, beam 4 | 123 | 27.98 | [tensorboard](https://tensorboard.dev/experiment/FViOeiMQS56qK4bzRvcvEA) | [weight](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_big_t2t_wmt2014_en_de_yttm_0.0006_2560_4_60_eps1e-9_20201226/avg_46_60.params) | [log](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_big_t2t_wmt2014_en_de_yttm_0.0006_2560_4_60_eps1e-9_20201226/train_transformer_rank0_local0_4.log) | [config](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_big_t2t_wmt2014_en_de_yttm_0.0006_2560_4_60_eps1e-9_20201226/config.yml) |

```
BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.14 = 28.0 58.8/33.6/21.5/14.4 (BP = 1.000 ratio = 1.021 hyp_len = 64018 ref_len = 62688)
```

#### Deep Encoder, Shallow Decoder

In addition, it has been shown that we can use a deep encoder and
a shallow decoder to improve the performance as in
["Deep Encoder, Shallow Decoder:Reevaluating the Speed-Quality Tradeoff in Machine Translation"](https://arxiv.org/pdf/2006.10369.pdf)
To train with Deep-Shallow architecture, you can change set the encoder and decoder layers as shown in [transformer_enc12_dec1.yml](transformer_enc12_dec1.yml).

```
SUBWORD_ALGO=yttm
SRC=en
TGT=de
lr=0.0006
num_accumulated=4
max_num_tokens=4096
wd=0.0
epochs=60
SAVE_DIR=transformer_base_${SRC}-${TGT}_enc12_dec1_${SUBWORD_ALGO}_${lr}_${wd}_${num_accumulated}_${max_num_tokens}_${epochs}
horovodrun -np 4 -H localhost:4 python3 train_transformer.py \
    --comm_backend horovod \
    --train_src_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${SRC} \
    --train_tgt_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${TGT} \
    --src_lang ${SRC} \
    --tgt_lang ${TGT} \
    --src_tokenizer ${SUBWORD_ALGO} \
    --tgt_tokenizer ${SUBWORD_ALGO} \
    --dev_src_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_tgt_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_tgt_raw_corpus wmt2014_ende/dev.raw.${TGT} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir ${SAVE_DIR} \
    --optimizer adam \
    --wd ${wd} \
    --cfg transformer_enc12_dec1.yml \
    --lr ${lr} \
    --num_accumulated ${num_accumulated} \
    --sampler BoundedBudgetSampler \
    --max_num_tokens ${max_num_tokens} \
    --epochs ${epochs} \
    --warmup_steps 4000 \
    --warmup_init_lr 1e-07 \
    --seed 123 \
    --max_grad_norm 1.0 \
    --fp16
```

We used the same evaluation method (by averaging the last 10 checkpoints) as in [Transformer Base](#transformer-base)

| Subword Model | Beam Search | Seed  | Test BLEU | Tensorboard | Weights | Log | Config |
|---------------|-------------|-------|-----------|-------------|---------|-----|--------|
| yttm          | lp_alpha 0.0, beam 4 | 123 | 26.25 | [tensorboard](https://tensorboard.dev/experiment/Xc1zsayHRYK0oOGhWJ3N5A) | [weight](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_base_en-de_enc12_dec1_yttm_0.0006_0.0_4_4096_60_20201229/avg_51_60.params) | [log](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_base_en-de_enc12_dec1_yttm_0.0006_0.0_4_4096_60_20201229/train_transformer_rank0_local0_4.log) | [config](https://gluon-nlp-log.s3.amazonaws.com/machine_translation/transformer_base_en-de_enc12_dec1_yttm_0.0006_0.0_4_4096_60_20201229/config.yml) |

```
BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.14 = 26.2 57.8/32.2/20.1/13.0 (BP = 0.994 ratio = 0.994 hyp_len = 62312 ref_len = 62688)
```
#### Stochastic Beam Search

Additionally, it has been shown that Stochastic Beam Search can obtain diverse translations with good quality in ["Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement"](https://arxiv.org/pdf/1903.06059.pdf). To use stochactic Beam Search for evaluation, add the ``--stochastic`` option and set the ``--temperature``. The default temperature is 0.5. According to the paper, Lower temperature can yield better translations. There are some results for reference.

|      | Temperature | Transforme_Base | Transformer_Big |
| ---- | ----------- | --------------- | --------------- |
| 1.   | 1.0         | 8.66            | 11.52           |
| 2.   | 0.5         | 24.22           | 25.19           |
| 3.   | 0.4         | 24.92           | 26.05           |
| 4.   | 0.3         | 25.46           | 26.58           |
| 5.   | 0.2         | 25.94           | 26.88           |
| 6.   | 0.1         | 26.28           | 27.26           |
| 7.   | 0.05        | 26.29           | 27.27           |
| 8.   | 0.01        | 26.32           | 27.22           |
