# Pretraining Corpus

We provide a series of shared scripts for downloading/preparing the text corpus for pretraining NLP models.
This helps create a unified text corpus for studying the performance of different pretraining algorithms.
When releasing the datasets, we follow the [FAIR principle](https://www.go-fair.org/fair-principles/),
i.e., the dataset needs to be findable, accessible, interoperable, and reusable.

## BookCorpus
Unfortunately, we are unable to provide the original [Toronto BookCorpus dataset](https://yknzhu.wixsite.com/mbweb) due to licensing issues.

There are some open source efforts for reproducing the dataset, e.g.,
 using [soskek/bookcorpus](https://github.com/soskek/bookcorpus) or directly downloading the [preprocessed version](https://drive.google.com/file/d/16KCjV9z_FHm8LgZw05RSuk4EsAWPOP_z/view).

Nevertheless, we utilize the [Project Gutenberg](https://www.gutenberg.org/) as an alternative to Toronto BookCorpus.

You can use the following command to download and prepare the Gutenberg dataset.

```bash
python3 prepare_bookcorpus.py --dataset gutenberg
```

Also, you should follow the [license](https://www.gutenberg.org/wiki/Gutenberg:The_Project_Gutenberg_License) for using the data.

## Wikipedia

Please install [attardi/wikiextractor](https://github.com/attardi/wikiextractor) for preparing the data.

```bash
# Download
python3 prepare_wikipedia.py --mode download --lang en --date latest -o ./

# Properly format the text files
python3 prepare_wikipedia.py --mode format -i [path-to-wiki.xml.bz2] -o ./

```
The process of downloading and formatting is time consuming, and we offer an alternative solution to download the prepared raw text file from S3 bucket. This raw text file is in English and was dumped at 2020-06-20 being formated by the above very process (` --lang en --date 20200620`).

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
