# Pretraining Corpus

We provide a series of shared scripts for downloading/preparing the text corpus for pretraining NLP models.
This helps create a unified text corpus for studying the performance of different pretraining algorithms.
When picking the datasets to support, we follow the [FAIR principle](https://www.go-fair.org/fair-principles/),
i.e., the dataset needs to be findable, accessible, interoperable, and reusable.

For all scripts, we can either use `nlp_data SCRIPT_NAME`, or directly call the script.

## Gutenberg BookCorpus
Unfortunately, we are unable to provide the [Toronto BookCorpus dataset](https://yknzhu.wixsite.com/mbweb) due to licensing issues.

There are some open source efforts for reproducing the dataset, e.g.,
 using [soskek/bookcorpus](https://github.com/soskek/bookcorpus) or directly downloading the [preprocessed version](https://drive.google.com/file/d/16KCjV9z_FHm8LgZw05RSuk4EsAWPOP_z/view).

Thus, we utilize the [Project Gutenberg](https://www.gutenberg.org/) as an alternative to Toronto BookCorpus.

You can use the following command to download and prepare the Gutenberg corpus.

```bash
python3 prepare_gutenberg.py --save_dir gutenberg
```

Also, you should follow the [license](https://www.gutenberg.org/wiki/Gutenberg:The_Project_Gutenberg_License) for using the data.

## Wikipedia

We used the [attardi/wikiextractor](https://github.com/attardi/wikiextractor) package for preparing the data.

The following is an example that downloads and formats the hindi wikicorpus.

```bash
# Download. 
# By default, it will download to wikicorpus/download/lang/date/wikicorpus.xml.bz2
python3 prepare_wikipedia.py --mode download --lang hi --date latest

# Properly format the text files
python3 prepare_wikipedia.py --mode format -i wikicorpus/download/hi/latest/wikicorpus.xml.bz2

# In addition, you may supress the logging by adding `--quiet`
python3 prepare_wikipedia.py --mode format -i wikicorpus/download/hi/latest/wikicorpus.xml.bz2 --quiet

```
After formatting, it will create a folder called `prepare_wikipedia`. The processed text files 
are chunked.

In addition, you can try to combine these two steps via
```
python3 prepare_wikipedia.py --mode download+format --lang hi --date latest --quiet
```

The process of downloading and formatting is time consuming, and we offer an alternative 
solution to download the prepared raw text file from S3 bucket. This raw text file is in English and 
was dumped at 2020-06-20 being formatted by the above process (` --lang en --date 20200620`).

```bash
python3 prepare_wikipedia.py --mode download_prepared -o ./
```
### References
- [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
- [attardi/wikiextractor](https://github.com/attardi/wikiextractor)

## OpenWebText

You can download the OpenWebText from [link](https://skylion007.github.io/OpenWebTextCorpus/).
After downloading and extracting the OpenWebText (i.e., `tar xf openwebtext.tar.xz`), you can use the following command to preprocess the dataset.

```bash
python3 prepare_openwebtext.py --input openwebtext/ --output prepared_owt --shuffle
```

In this step, the archived txt are directly read without decompressing.
They are concatenated together in a single txt file with the same name as the archived file, using double empty lines as the document separation.


## BookCorpus

You can use the following instruction to download BookCorpus dataset.

```bash
python3 prepare_bookcorpus.py  --output BookCorpus 
```

And you can get raw text files (one article per text file) and
formulation 1(one article per line in one text file called bookcorpus.txt) 

For using of SOP/NSP loss in training BERT, you can get formulation 2 (one sentence per line) by

```bash
python3 prepare_bookcorpus.py  --output BookCorpus --segment_sentences --segment_num_worker 16
```


