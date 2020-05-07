# Question Answering

## SQuAD

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


## Natural Questions
```bash
gsutil -m cp -R gs://natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz .
```
