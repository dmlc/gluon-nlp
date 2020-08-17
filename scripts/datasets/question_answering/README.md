# Question Answering

## SQuAD
SQuAD datasets is distributed under the [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/legalcode) license.

Run the following command to download squad

```bash
python3 prepare_squad.py --version 1.1 # Squad 1.1
python3 prepare_squad.py --version 2.0 # Squad 2.0
```

For all datasets we support, we provide command-line-toolkits for downloading them as

```bash
nlp_data prepare_squad --version 1.1
nlp_data prepare_squad --version 2.0
```

Directory structure of the squad dataset will be as follows, where `version` can be 1.1 or 2.0:
```
squad
├── train-v{version}.json
├── dev-v{version}.json
```

## SearchQA
Following BSD-3-Clause License, we uploaded the SearchQA to our S3 bucket and provide the link to download the processed txt files. Please check out the [Google drive link](https://drive.google.com/drive/u/0/folders/1kBkQGooNyG0h8waaOJpgdGtOnlb1S649) to download to raw and split files collected through web search using the scraper from [GitHub repository](https://github.com/nyu-dl/dl4ir-searchQA).

Download SearchQA Dataset with python command or Command-line Toolkits

```bash
python3 prepare_searchqa.py

# Or download with command-line toolkits
nlp_data prepare_searchqa
```

Directory structure of the searchqa dataset will be as follows
```
searchqa
├── train.txt
├── val.txt
├── test.txt
```

## TriviaQA
[TriviaQA](https://nlp.cs.washington.edu/triviaqa/) is an open domain QA dataset. See more useful scripts in [Offical Github](https://github.com/mandarjoshi90/triviaqa)

Run the following command to download triviaqa

```bash
python3 prepare_triviaqa.py --version rc         # Download TriviaQA version 1.0 for RC (2.5G)
python3 prepare_triviaqa.py --version unfiltered # Download unfiltered TriviaQA version 1.0 (604M)

# Or download with command-line toolkits
nlp_data prepare_triviaqa --version rc
nlp_data prepare_triviaqa --version unfiltered
```

Directory structure of the triviaqa (rc and unfiltered) dataset will be as follows:
```
triviaqa
├── triviaqa-rc
    ├── qa
        ├── verified-web-dev.json        
        ├── web-dev.json                   
        ├── web-train.json     
        ├── web-test-without-answers.json
        ├── verified-wikipedia-dev.json
        ├── wikipedia-test-without-answers.json
        ├── wikipedia-dev.json  
        ├── wikipedia-train.json
    ├── evidence
        ├── web
        ├── wikipedia

├── triviaqa-unfiltered
    ├── unfiltered-web-train.json
    ├── unfiltered-web-dev.json
    ├── unfiltered-web-test-without-answers.json
```

## HotpotQA
HotpotQA is distributed under a [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/). We only provide download scripts (run by the following command), and please check out the [GitHub repository](https://github.com/hotpotqa/hotpot) for the details of preprocessing and evaluation.

```bash
python3 prepare_hotpotqa.py

# Or download with command-line toolkits
nlp_data prepare_hotpotqa
```

Directory structure of the hotpotqa dataset will be as follows
```
hotpotqa
├── hotpot_train_v1.1.json
├── hotpot_dev_fullwiki_v1.json
├── hotpot_dev_distractor_v1.json
├── hotpot_test_fullwiki_v1.json
```
