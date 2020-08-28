# Language Understanding Benchmarks

We provide the documentation about how to download and prepare the 
[GLUE](https://gluebenchmark.com/) and [SuperGLUE](https://super.gluebenchmark.com/).

These benchmarks share the common goal of providing a robust set of downstream tasks for evaluating 
the NLP models' performance.

In essence, these NLP tasks share a similar structure. We are interested in the question: 
can we design a model that can solve these tasks all in once? 
[BERT](https://arxiv.org/pdf/1810.04805.pdf) has done a good job in unifying the way to 
featurize the text data, in which we extract two types of embeddings: one for the 
whole sentence and the other for each tokens in the sentence. Later, 
in [T5](https://arxiv.org/pdf/1910.10683.pdf), the author proposed to convert every task 
into a text-to-text problem. However, it is difficult to convert tasks like measuring the similarity between sentences,
or named-entity recognition into text-to-text, because they involve real-values or entity
spans that are difficult to be encoded as raw text data.

In GluonNLP, we propose a unified way to tackle these NLP problems. We convert these datasets 
as tables. Each column in the table will be 1) raw text, 2) entity/list of entities associated with the 
raw text, 3) numerical values or a list of numerical values. 
In addition, we keep a metadata object that describes 1) the relationship among columns, 
2) certain properties of the columns.

All tasks used in these general benchmarks are converted to this format.


## GLUE Benchmark

The details of the benchmark are described in [GLUE Paper](https://openreview.net/pdf?id=rJ4km2R5t7).

To obtain the datasets, run:

```
nlp_data prepare_glue --benchmark glue
```

There will be multiple task folders. All data are converted into pandas dataframes + an additional 
`metadata.json` object.

Here are the details of the datasets:

| Dataset | #Train | #Dev | #Test   | Columns         | Task                         | Metrics                      | Domain              |
|---------|--------|------|--------|---------------------|------------------------------|------------------------------|---------------------|
| CoLA    | 8.5k   | 1k   | 1k     | sentence, **label**  | acceptability  (0 / 1)       | Matthews corr.               | misc.               |
| SST-2   | 67k    | 872  | 1.8k   | sentence, **label**     | sentiment                    | acc.                         | movie reviews       |
| MRPC    | 3.7k   | 408  | 1.7k   | sentence1, sentence2, **label** | paraphrase                   | acc./F1                      | news                |
| STS-B   | 5.7k   | 1.5k | 1.4k   | sentence1, sentence2, **score** | sentence similarity          | Pearson/Spearman corr.       | misc.                |
| QQP     | 364k   | 40k  | 391k   | sentence1, sentence2, **label** | paraphrase                   | acc./F1                      | social QA questions |
| MNLI    | 393k   | 9.8k(m) / 9.8k(mm)  | 9.8k(m) / 9.8k(mm)  | sentence1, sentence2, genre, **label** | NLI    | matched acc./mismatched acc. | misc                |
| QNLI    | 105k   | 5.4k | 5.4k   | question, sentence, **label** | QA/NLI                       | acc.                         | Wikipedia           |
| RTE     | 2.5k   | 227  | 3k     | sentence1, sentence2, **label** | NLI                          | acc.                         | news, Wikipedia     |
| WNLI    | 634    |  71  | 146    | sentence1, sentence2, **label** | NLI                          | acc.                         | fiction books       |

In addition, GLUE has the diagnostic task that tries to analyze the system's performance on a broad range of linguistic phenomena. 
It is best described in [GLUE Diagnostic](https://gluebenchmark.com/diagnostics). 
The diagnostic dataset is based on Natural Language Inference (NLI) and you will need to use the model trained with 
MNLI on this dataset.

| Dataset     | #Sample | Data Format | Metrics         |
|-------------|---------|-------------|-----------------|
| Diagnostic  | 1104    | semantics, predicate, logic, knowledge, domain, premise, hypothesis, label | Matthews corr.  |

In addition, we provide the SNLI dataset, which is recommend as an auxiliary data source for training MNLI. 
This is the recommended approach in [GLUE](https://openreview.net/pdf?id=rJ4km2R5t7).

| Dataset | #Train  | #Test  | Data Format                 | Task | Metrics | Domain |
|---------|---------|--------|-----------------------------|------|---------|--------|
| SNLI    | 549K    | 20k    | sentence1, sentence2, **label** | NLI  | acc.    | misc   |


## SuperGLUE Benchmark

The details are described in [SuperGLUE Paper](https://arxiv.org/pdf/1905.00537.pdf).

To obtain the datasets, run:

```
nlp_data prepare_glue --benchmark superglue
```


| Dataset  | #Train  | #Dev | #Test   | Columns         | Task         | Metrics                      | Domain                          |
|----------|---------|------|---------|---------------------|--------------|------------------------------|---------------------------------|
| BoolQ    | 9.4k    | 3.3k | 3.2k    | passage, question, **label** | QA           | acc.                         | Google queries, Wikipedia       |
| CB       | 250     | 57   | 250     | premise, hypothesis, **label** | NLI          | acc./F1                      | various                         |
| COPA     | 400     | 100  | 500     | premise, choice1, choice2, question, **label** | QA           | acc.                         | blogs, photography encyclopedia |
| MultiRC* | 5.1k (27k)  | 953 (4.8k) | 1.8k (9.7k) | passage, question, answer, **label**                  | QA           | F1/EM                        | various                         |
| ReCoRD   | 101k    | 10k  | 10k         | source, text, entities, query, **answers** | QA           | F1/EM                        | news                            |
| RTE      | 2.5k    | 278  | 3k          | premise, hypothesis, **label**  | NLI          | acc.                         | news, Wikipedia                 |
| WiC      | 6k    | 638  | 1.4k          | sentence1, sentence2, entities1, entities2, **label**  | WSD          | acc.                         | WordNet, VerbNet, Wiktionary    |
| WSC      | 554     | 104  | 146         | text, entities, **label**  | coref.       | acc.                         | fiction books                   |

*Note that for MultiRC, we enumerated all combinations of (passage, question, answer) triplets in 
the dataset and the number of samples in the expanded format is recorded inside parenthesis.

Similar to GLUE, SuperGLUE has two diagnostic tasks to analyze the system performance 
on a broad range of linguistic phenomena. For more details, 
see [SuperGLUE Diagnostic](https://super.gluebenchmark.com/diagnostics).

| Dataset       | #Samples | Columns                        |Metrics         |
|---------------|----------|----------------------|----------------|
| Winogender    | 356 |hypothesis, premise, label | Accuracy       |
| Broadcoverage | 1104  | label, sentence1, sentence2, logic | Matthews corr. |

## Text Classification Benchmark

We also provide the script to download a series of text classification datasets for the purpose of 
benchmarking. We select the classical datasets that are also used in 
[Character-level Convolutional Networks for TextClassification, NeurIPS2015](https://arxiv.org/pdf/1509.01626.pdf)
 and [Funnel-Transformer: Filtering out SequentialRedundancy for Efficient Language Processing, Arxiv2020](https://arxiv.org/pdf/2006.03236.pdf). 

| Dataset       | #Train  | #Test   | Columns         | Metrics         |
|---------------|---------|---------|-----------------|-----------------|
| AG            | 120000  | 7600    | content, label  | acc             |
| IMDB          | 25000   | 25000   | content, label  | acc             |
| DBpedia       | 560000  | 70000   | content, label  | acc             |
| Yelp2         | 560000  | 38000   | content, label  | acc             |
| Yelp5         | 650000  | 50000   | content, label  | acc             |
| Amazon2       | 3600000 | 400000  | content, label  | acc             |
| Amazon5       | 3000000 | 650000  | content, label  | acc             |

To obtain the datasets, run:

```
nlp_data prepare_text_classification -t all
```
