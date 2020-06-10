# Machine Translation

## Train a Transformer from scratch
First, use the script described in [datasets/machine_translation](../datasets/machine_translation) 
to generate the dataset. Then, run `train_transformer.py` to train the model. 

In the following, we give the training script for WMT2014 EN-DE task with yttm tokenizer. 
You may first run the following command in [datasets/machine_translation](../datasets/machine_translation).
```bash
bash wmt2014_ende.sh yttm
```

Then, you can run the experiment, we use the
"transformer_nmt_base" configuration.

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
    --cfg transformer_nmt_base \
    --lr 0.002 \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 123 \
    --gpus 0,1,2,3
```

Use the following command to inference/evaluate the Transformer model:

```bash
SUBWORD_MODEL=yttm
python evaluate_transformer.py \
    --param_path transformer_wmt2014_ende_${SUBWORD_MODEL}/average.params \
    --src_lang en \
    --tgt_lang de \
    --cfg transformer_wmt2014_ende_${SUBWORD_MODEL}/config.yml \
    --src_tokenizer ${SUBWORD_MODEL} \
    --tgt_tokenizer ${SUBWORD_MODEL} \
    --src_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --tgt_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --src_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --tgt_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --src_corpus ../datasets/machine_translation/wmt2014_ende/test.raw.en \
    --tgt_corpus ../datasets/machine_translation/wmt2014_ende/test.raw.de
```


Test BLEU score with 3 seeds (evaluated via sacre BLEU):

- transformer_nmt_base

| Subword Model | #Params    | Seed = 123  | Seed = 1234 | Seed = 12345 |  Mean±std   |
|---------------|------------|-------------|-------------|--------------|-------------|
| yttm          |            |  26.63      | 26.73       |              |  -          |
| hf_bpe        |            |  -          | -           |  -           |  -          |
| spm           |            |  -          | -           |  -           |  -          |
