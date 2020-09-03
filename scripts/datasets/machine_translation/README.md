# Machine Translation

In machine translation, we train a model to map a sentence from the source language, e.g., English, 
to the target language, e.g., Chinese. Here, we provide scripts to download the common benchmark 
datasets for machine translation. The downloaded datasets are stored as a pair of corpus files, 
one for the source and the other for the target.  

## WMT
You can use [prepare_wmt.py](prepare_wmt.py) to download and prepare the raw training corpus and 
then use [clean_parallel_corpus.py](../../preprocess/clean_parallel_corpus.py) to clean and 
filter the corpus. 

You may download the raw WMT2014 en-de  
```bash
nlp_data prepare_wmt \
        --dataset wmt2014 \
        --lang-pair en-de \
        --save-path wmt2014_en_de
```

By combining `nlp_data` and `nlp_process`, we provide the example for preparing the 
WMT2014 en-de training dataset: [wmt2014_ende.sh](wmt2014_ende.sh). This involves three steps:
- Downloading the raw text data
- Clean and tokenize the data
- Learn subword model and apply the learned subword model.

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


Apart from WMT2014 EN-DE, we also provided the script for preparing the training data for 
WMT2017 ZH-EN task: 
[wmt2017_zhen.sh](wmt2017_zhen.sh).

### Monolingual Corpus
In the WMT competition, there are additional monolingual corpus that helps you train NMT models. 
You may download the raw monolingual corpus by adding `--mono` flag.

One example is to download the newscrawl monolingual corpus in German:

```bash
nlp_data prepare_wmt \
        --mono \
        --mono_lang de \
        --dataset newscrawl \
        --save-path wmt2014_mono
```   


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
