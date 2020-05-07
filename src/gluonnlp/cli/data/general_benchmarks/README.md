# General Benchmarks for NLP

We provide the tools for downloading and preparing the 
[GLUE](https://gluebenchmark.com/), [SuperGLUE](https://super.gluebenchmark.com/),
 and [SentEval](https://www.aclweb.org/anthology/L18-1269.pdf) benchmarks.

These benchmarks share the common goal of providing a robust set of downstream tasks for evaluating the NLP models' performance.

## GLUE Benchmark

The details are described in [GLUE Paper](https://openreview.net/pdf?id=rJ4km2R5t7).

| Dataset                         | #Train | #Test   | Data Format         | Task                         | Metrics                      | Domain              |
|---------------------------------|---------|--------|---------------------|------------------------------|------------------------------|---------------------|
| CoLA                            | 8.5k    | 1k     | Text, Score         | acceptability                | Matthews corr.               | misc.               |
| SST-2                           | 67k     | 1.8k   | Text, Label         | sentiment                    | acc.                         | movie reviews       |
| MRPC                            | 3.7k    | 1.7k   | TextA, TextB, Label | paraphrase                   | acc./F1                      | news                |
| STS-B                           | 7k      | 1.4k   | TextA, TextB, Score | sentence similarity          | Pearson/Spearman corr.       | misc.                |
| QQP                             | 364k    | 391k   | TextA, TextB, Label | paraphrase                   | acc./F1                      | social QA questions |
| MNLI                            | 393k    | 20k    | TextA, TextB, Label | NLI                          | matched acc./mismatched acc. | misc                |
| QNLI                            | 105k    | 5.4k   | TextA, TextB, Label | QA/NLI                       | acc.                         | Wikipedia           |
| RTE                             | 2.5k    | 3k     | TextA, TextB, Label | NLI                          | acc.                         | news, Wikipedia     |
| WNLI                            | 634     | 146    | TextA, TextB, Label | NLI                          | acc.                         | fiction books       |

In addition, GLUE has the diagnostic task that tries to analyze the system's performance on a broad range of linguistic phenomena. 
It is best described in [GLUE Diagnostic](https://gluebenchmark.com/diagnostics). 
The diagnostic dataset is based on Natural Language Inference (NLI) and you will need to use the model trained with 
MNLI on this dataset.

| Dataset       | #Samples  | Data Format     | Metrics         |
|---------------|-----------|-----------------|-----------------|
| Diagnostic    | 1104      | TextA, TextB,   | Matthews corr.  |


## SuperGLUE Benchmark

The details are described in [SuperGLUE Paper](https://arxiv.org/pdf/1905.00537.pdf).


| Dataset                         | #Train  | #Dev | #Test   | Data Format         | Task         | Metrics                      | Domain                          |
|---------------------------------|---------|------|---------|---------------------|--------------|------------------------------|---------------------------------|
| BoolQ                           | 9427    | 3270 | 3245    | Text, Score         | QA           | acc.                         | Google queries, Wikipedia       |
| CB                              | 250     | 57   | 250     | Text, Label         | NLI          | acc./F1                      | various                         |
| COPA                            | 400     | 100  | 500     | TextA, TextB, Label | QA           | acc.                         | blogs, photography encyclopedia |
| MultiRC                         | 5100    | 953  | 1800    | TextA, TextB, Score | QA           | F1/EM                        | various                         |
| ReCoRD                          | 101k    | 10k  | 10k     | TextA, TextB, Label | QA           | F1/EM                        | news                            |
| RTE                             | 2500    | 278  | 300     | TextA, TextB, Label | NLI          | acc.                         | news, Wikipedia                 |
| WiC                             | 6000    | 638  | 1400    | TextA, TextB, Label | WSD          | acc.                         | WordNet, VerbNet, Wiktionary    |
| WSC                             | 554     | 104  | 146     | TextA, TextB, Label | coref.       | acc.                         | fiction books                   |

Similar to GLUE, SuperGLUE has two diagnostic tasks to analyze the system performance 
on a broad range of linguistic phenomena. For more details, 
see [SuperGLUE Diagnostic](https://super.gluebenchmark.com/diagnostics).

| Dataset       | #Samples  | Data Format     | Metrics         |
|---------------|-----------|-----------------|-----------------|
| Diagnostic    | 1104      | TextA, TextB,   | Matthews corr.  |

## SentEval Benchmark

The details are described in [SentEval](https://github.com/facebookresearch/SentEval).
