# Machine Translation

In machine translation, we train a model to map a sentence from the source language, e.g., English, 
to the target language, e.g., Chinese. Here, we provide scripts to download the common benchmark 
datasets for machine translation. The downloaded datasets are stored as a pair of corpus files, 
one for the source and the other for the target.  

## WMT
You can use [prepare_wmt.py](prepare_wmt.py) to download and prepare the raw training corpus and 
then use [clean_parallel_corpus.py](../../preprocess/clean_parallel_corpus.py) to clean and 
filter the corpus. 

For example, to prepare the WMT2014 en-de dataset, we can use the command described in 
[wmt2014_ende.sh](wmt2014_ende.sh).

```bash
bash wmt2014_ende.sh yttm
```

We support the following subword learning algorithms:

```bash
# BPE from YouTokenToMe
bash wmt2014_ende.sh yttm

# BPE from Huggingface
bash wmt2014_ende.sh hf_bpe

# BPE from subword-nmt
bash wmt2014_ende.sh subword_nmt

# Byte-level BPE
bash wmt2014_ende.sh hf_bytebpe

# Sentencepiece
bash wmt2014_ende.sh spm

# WordPiece
bash wmt2014_ende.sh hf_wordpiece
```


To prepare the WMT2017 zh-en dataset, we can use the command described in 
[wmt2017_zhen.sh](wmt2017_zhen.sh).


### Directory Structure of Translation Dataset

The basic structure of a translation dataset is like the following:
```
folder_name
├── train.raw.{src}
├── train.raw.{tgt}
├── train.tok.{src}
├── train.tok.{tgt}
├── train.tok.{subword_model}.{src}
├── train.tok.{subword_model}.{tgt}
├── ... 
├── ... Repeat for valid and test
├── ...
├── {subword_model}.model
├── {subword_model}.path
```

## IWSLT
TBA
