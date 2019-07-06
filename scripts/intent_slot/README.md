# Joint Intent Classification and Slot Labeling by GluonNLP


## Introduction
Intent classification and slot labeling are two essential problems in Natural Language Understanding (NLU).
In _intent classification_, the agent needs to detect the intention that the speaker's utterance conveys.
 For example, when the speaker says "Book a flight from Long Beach to Seattle", the intention is to book a flight ticket.
In _slot labeling_, the agent needs to extract the semantic entities that are related to the intent. In our previous example,
"Long Beach" and "Seattle" are two semantic constituents related to the flight, i.e., the origin and the destination.

Essentially, _intent classification_ can be viewed as a sequence classification problem and _slot labeling_ can be viewed as a
sequence tagging problem similar to Named-entity Recognition (NER). Due to their inner correlation, these two tasks are usually
trained jointly with a multi-task objective function.  

Here's one example of the ATIS dataset

| Sentence  | Tags | Intent Label |
| --------- | ---- | ------------ |
|    are    | O    |    atis_flight |
| there     | O    |  |
| any       | O    |  |
| flights   | O    |  |
| from      | O    |  |
| long      | B-fromloc.city_name |  |
| beach     | I-fromloc.city_name |  |
| to        | O                   |  |
| columbus  | B-toloc.city_name   |  |
| on        | O                   |  |
| wednesday | B-depart_date.day_name    |  |
| april     | B-depart_date.month_name  |  |
| sixty     | B-depart_date.day_number  |  |



In this example, we demonstrate how to use GluonNLP to build a model to perform joint intent classification and slot labeling. We 
choose to finetune a pretrained BERT model.  We use two datasets [ATIS](https://github.com/yvchen/JointSLU) and [SNIPS](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines).
 
## Requirements

```
mxnet
gluonnlp
seqeval
```

You may use pip or other tools to install these packages

## Experiment
For the ATIS dataset, use the following command to run the experiment:
```bash
python demo.py --gpu 0 --dataset atis
```

It produces the final slot labeling F1 = `95.83%` and intent classification accuracy = `98.66%`

For the SNIPS dataset, use the following command to run the experiment:
```bash
python demo.py --gpu 0 --dataset snips
```
It produces the final slot labeling F1 = `95.76%` and intent classification accuracy = `98.71%`

Also, we train the models with three random seeds and report the mean/std

For ATIS

| Models | Intent Acc (%) | Slot F1 (%) |
| ------ | ------------------------ | ----------- |
| [Intent Gating & self-attention, EMNLP 2018](https://www.aclweb.org/anthology/D18-1417) | 98.77 | 96.52 |
| [BLSTM-CRF + ELMo, AAAI 2019](https://arxiv.org/abs/1811.05370) | 97.42 | 95.62 |
| [Joint BERT, Arxiv 2019](https://arxiv.org/pdf/1902.10909.pdf) |  97.5 | 96.1 |
| Ours | 98.66±0.00  | 95.88±0.04 |

For SNIPS

| Models | Intent Acc (%) | Slot F1 (%) |
| ------ | ------------------------ | ----------- |
| [BLSTM-CRF + ELMo, AAAI 2019](https://arxiv.org/abs/1811.05370) | 99.29 | 93.90 |
| [Joint BERT, Arxiv 2019](https://arxiv.org/pdf/1902.10909.pdf) | 98.60 | 97.00 |
| Ours | 98.81±0.13 | 95.94±0.10 |
