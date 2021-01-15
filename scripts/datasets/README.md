# Datasets

This page describes how to download and prepare the datasets used in GluonNLP.

Essentially, we provide scripts for downloading and preparing the datasets.
The directory structure and the format of the processed datasets are well documented so that you are able to
reuse the scripts with your own data (as long as the structure/format matches).

Thus, the typical workflow for running experiments:

- Download and prepare data with scripts in [datasets](.).
- In case you will need to preprocess the dataset, there are toolkits in [preprocess](../preprocess).
- Run the experiments in [scripts](..)


## Available Datasets

- [Machine Translation](./machine_translation)
    - [WMT](./machine_translation/README.md#wmt)
- [Question Answering](./question_answering)
    - [SQuAD](./question_answering/README.md#squad)
    - [SearchQA](./question_answering/README.md#searchqa)
    - [TriviaQA](./question_answering/README.md#triviaqa)
    - [HotpotQA](./question_answering/README.md#hotpotqa)
    - [NaturalQuestions](./question_answering/README.md#NaturalQuestions)

- [Language Modeling](./language_modeling)
    - [WikiText-2](./language_modeling)
    - [WikiText-103](./language_modeling)
    - [Text8](./language_modeling)
    - [Enwiki8](./language_modeling)
    - [Google Billion Words](./language_modeling)
- [Music Generation](./music_generation)
    - [LakhMIDI](./music_generation/README.md#lakh-midi)
    - [MAESTRO](./music_generation/README.md#maestro)
- [Pretraining Corpus](./pretrain_corpus)
    - [Wikipedia](./pretrain_corpus/README.md#wikipedia)
    - [Gutenberg BookCorpus](./pretrain_corpus/README.md#gutenberg-bookcorpus)
    - [OpenWebText](./pretrain_corpus/README.md#openwebtext)
- [General NLP Benchmarks](./general_nlp_benchmark)
    - [GLUE](./general_nlp_benchmark/README.md#glue-benchmark)
    - [SuperGLUE](./general_nlp_benchmark/README.md#superglue-benchmark)
    - [Text Classification Benchmark](./general_nlp_benchmark/README.md#text-classification-benchmark)

## Contribution Guide

We are very happy to receive and merge your contributions about new datasets :smiley:.

To add a new dataset, you may create a `prepare_{DATASET_NAME}.py` file in the specific folder.
Also, remember to add the documentation in the `README.md` about 1) the directory structure and 2) how to use the CLI tool for downloading + preprocessing.
In addition, add citations in the `prepare_{DATASET_NAME}.py` to assign credit to the original author.
Refer to the existing scripts or ask questions in Github if you need help.  

All URLs are bound with SHA1-hash keys to make sure that the downloaded files are not corrupted. You can refer to the files in [url_checksums](./url_checksums) for examples.

In order to generate the hash values of the data files, you can revise [update_download_stats.py](update_download_stats.py)
and include the new URLS + create the stats file that will store the hash keys. Use the following command to update the hash key:

```bash
python3 update_download_stats.py
```

## Frequently Asked Questions
- After installing GluonNLP, I cannot access the command line toolkits. It reports `nlp_data: command not found`.
  
  The reason is that you have installed glunonnlp to a folder that is not in `PATH`, e.g.,  
  `~/.local/bin`. You can try to change the `PATH` variable to also include '~/.local/bin' via the following command:
  
  ```
  export PATH=${PATH}:~/.local/bin
  ```
