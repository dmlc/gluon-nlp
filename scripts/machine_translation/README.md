# Machine Translation

## Transformers

### Training with your own data
First, use the script described in [datasets/machine_translation](../datasets/machine_translation) 
to generate the dataset. Then, run `train_transformer.py` to train the model. 

In the following, we give the training script for WMT2014_EN-DE with yttm tokenizer. 
You may first run the following command in [datasets/machine_translation](../datasets/machine_translation).
```bash
bash wmt2014_ende.sh yttm
```

```bash
SUBWORD_MODEL=yttm
python train_transformer.py \
    --train_src_corpus ../datasets/machine_translation/wmt2014_ende/train.tok.${SUBWORD_MODEL}.en \
    --train_tgt_corpus ../datasets/machine_translation/wmt2014_ende/train.tok.${SUBWORD_MODEL}.de \
    --dev_src_corpus ../datasets/machine_translation/wmt2014_ende/dev.tok.${SUBWORD_MODEL}.en \
    --dev_tgt_corpus ../datasets/machine_translation/wmt2014_ende/dev.tok.${SUBWORD_MODEL}.de \
    --src_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --src_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --tgt_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --tgt_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --save_dir transformer_wmt2014_ende_${SUBWORD_MODEL} \
    --cfg wmt_en_de_base.yml \
    --lr 0.002 \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 100 \
    --gpus 0,1,2,3
```

Use the following command to evaluate the Transformer model 
(You may see some warnings and we will fix it later.):

```bash
SUBWORD_MODEL=yttm
python evaluate_transformer.py \
    --param_path transformer_wmt2014_ende_${SUBWORD_MODEL}/average.params \
    --src_lang en \
    --tgt_lang de \
    --cfg wmt_en_de_base.yml \
    --src_tokenizer ${SUBWORD_MODEL} \
    --tgt_tokenizer ${SUBWORD_MODEL} \
    --src_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --tgt_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --src_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --tgt_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --src_corpus ../datasets/machine_translation/wmt2014_ende/test.raw.en \
    --tgt_corpus ../datasets/machine_translation/wmt2014_ende/test.raw.de
```


Test BLEU score with 3 seeds:

| Seed = 100 | Seed = 1234 | Seed = 12345 |  Mean±std   |
| ---------- | ----------- | ------------ |  ---------- |
|   26.61    |   -    |   -          |  -          | 


### Back-Translation
