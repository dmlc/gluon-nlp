# Question Answering

## SQuAD
SQuAD datasets is distributed under the [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/legalcode) license.

Run the following command to download squad

```bash
python prepare_squad.py --version 1.1 # Squad 1.1
python prepare_squad.py --version 2.0 # Squad 2.0
```

Directory structure of the squad dataset will be as follows, where `version` can be 1.1 or 2.0:
```
squad
├── train-v{version}.json
├── dev-v{version}.json
```

## SearchQA

```bash
python prepare_searchqa.py
```

## Natural Questions

## TriviaQA

## HotpotQA
HotpotQA is distributed under a [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/). We only provide download scripts, and please check out the [GitHub repository](https://github.com/hotpotqa/hotpot) for the details of preprocessing and evaluation.

Run the following command to download hotpotqa

```bash
python prepare_hotpotqa.py
```

Directory structure of the hotpotqa dataset will be as follows
```
hotpotqa
├── hotpot_train_v1.1.json
├── hotpot_dev_fullwiki_v1.json
├── hotpot_dev_distractor_v1.json
├── hotpot_test_fullwiki_v1.json
```
## NewsQA
