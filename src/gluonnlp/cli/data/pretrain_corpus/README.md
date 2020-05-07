# Pretraining Corpus

We provide a series of shared scripts for downloading/preparing the text corpus for pretraining NLP models.
This helps create a unified text corpus for studying the performance of different pretraining algorithms.

## BookCorpus
Unfortunately, we are unable to provide the original [Toronto BookCorpus dataset](https://yknzhu.wixsite.com/mbweb) due to licensing issues.

There are some open source efforts for reproducing the dataset, e.g.,
 using [soskek/bookcorpus](https://github.com/soskek/bookcorpus) or directly downloading the [preprocessed version](https://drive.google.com/file/d/16KCjV9z_FHm8LgZw05RSuk4EsAWPOP_z/view). 
 
Nevertheless, we utilize the [Project Gutenberg](https://www.gutenberg.org/) as an alternative to Toronto BookCorpus.

You can use the following command to download the Gutenberg dataset. 

```bash
python prepare_bookcorpus.py --dataset gutenberg
```

Also, you should follow the [license](https://www.gutenberg.org/wiki/Gutenberg:The_Project_Gutenberg_License) for using the data. 

## Wikipedia

```
# Download
python prepare_wikipedia.py --mode download --lang en --date latest -o ./

# Properly format the text files
python prepare_wikipedia.py --mode format -i [path-to-wiki.xml.bz2] -o ./

```
### References
- [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
- [attardi/wikiextractor](https://github.com/attardi/wikiextractor)

## OpenWebText

You can download the OpenWebText from [link](https://skylion007.github.io/OpenWebTextCorpus/).
After downloading and extracting the OpenWebText, you can use the following command to preprocess 
the data for pretraining purpose.
(TBA)

# CC-News

(TBA)

# C4
(TBA)
