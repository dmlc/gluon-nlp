# Datasets

This page describes how to download and prepare the datasets used in GluonNLP.

Essentially, we provide scripts for downloading and preparing the datasets. 
The directory structure and the format of the processed datasets are well documented so that you are able to 
reuse the scripts with your own data (as long as the structure/format matches).

Thus, the typical workflow for running experiments:

- Download and prepare data with scripts in [datasets](.).
In case you will need to preprocess the dataset, there are toolkits in [preprocess](../preprocess).
- Run the experiments in [scripts](../scripts)


## Available Datasets
- [Machine Translation](./machine_translation)
    - [WMT](./machine_translation/README.md#wmt)
    - [IWSLT](./machine_translation/README.md#iwslt)
- [Question Answering](./question_answering)
    - [SQuAD](./question_answering/README.md#squad)
    - [Natural Questions](TBA)
- [Language Modeling](./language_modeling)
    - [WikiText-2](./language_modeling)
    - [WikiText-103](./language_modeling)
    - [Text8](./language_modeling)
    - [Enwiki8](./language_modeling)
    - [Google Billion Words](./language_modeling)
- [Conversational AI](./conversations)
    - [Intent Classification and Slot Labeling](TBA)
- [Sentiment Analysis](TBA)
    - [IMDB](TBA)
- [Music Generation](TBA)
    - [LakhMIDI](./music_generation/README.md#lakh-midi)
    - [MAESTRO](./music_generation/README.md#maestro)
- [Pretraining Corpus](./pretrain_corpus)
    - [Wikipedia](TBA)
    - [BookCorpus](TBA)
    - [OpenWebText](TBA)
- [General NLP Benchmarks](./general_benchmarks)
    - [GLUE](./general_benchmarks/README.md#glue-benchmark)
    - [SuperGLUE](./general_benchmarks/README.md#superglue-benchmark)
    - [SentEval](./general_benchmarks/README.md#senteval-benchmark)

## Contribution Guide

**TODO(sxjscience) Move to another contribution page + Add template for contribution**

We are very happy to receive and merge your contributions about new datasets :smiley:.

To add a new dataset, you may create a `prepare_{DATASET_NAME}.py` file in the specific folder.
Also, remember to add the documentation in the `README.md` about 1) the directory structure and 2) how to use the CLI tool for downloading + preprocessing.
In addition, add citations in the `prepare_{DATASET_NAME}.py` to assign credit to the original author. 
Refer to the existing scripts or ask questions in Github if you need help.  

All URLs are bound with SHA1-hash keys to make sure that the downloaded files are not corrupted. You can refer to the files in [url_checksums](./url_checksums) for examples.
 
In order to generate the hash values of the data files, you can revise [update_download_stats.py](update_download_stats.py) 
and include the new URLS + create the stats file that will store the hash keys. Use the following command to update the hash key:

```bash
python update_download_stats.py
```
