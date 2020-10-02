Datasets
========

This page describes how to download and prepare the datasets used in
GluonNLP.

Essentially, we provide scripts for downloading and preparing the
datasets. The directory structure and the format of the processed
datasets are well documented so that you are able to reuse the scripts
with your own data (as long as the structure/format matches).

Thus, the typical workflow for running experiments:

-  Download and prepare data with scripts in `datasets <.>`__.
-  In case you will need to preprocess the dataset, there are toolkits
   in `preprocess <../preprocess>`__.
-  Run the experiments in `scripts <..>`__

Available Datasets
------------------

-  `Machine Translation <./machine_translation>`__

   -  `WMT <./machine_translation/README.md#wmt>`__

-  `Question Answering <./question_answering>`__

   -  `SQuAD <./question_answering/README.md#squad>`__
   -  `SearchQA <./question_answering/README.md#searchqa>`__
   -  `TriviaQA <./question_answering/README.md#triviaqa>`__
   -  `HotpotQA <./question_answering/README.md#hotpotqa>`__

-  `Language Modeling <./language_modeling>`__

   -  `WikiText-2 <./language_modeling>`__
   -  `WikiText-103 <./language_modeling>`__
   -  `Text8 <./language_modeling>`__
   -  `Enwiki8 <./language_modeling>`__
   -  `Google Billion Words <./language_modeling>`__

-  `Music Generation <./music_generation>`__

   -  `LakhMIDI <./music_generation/README.md#lakh-midi>`__
   -  `MAESTRO <./music_generation/README.md#maestro>`__

-  `Pretraining Corpus <./pretrain_corpus>`__

   -  `Wikipedia <./pretrain_corpus/README.md#wikipedia>`__
   -  `Gutenberg
      BookCorpus <./pretrain_corpus/README.md#gutenberg-bookcorpus>`__
   -  `OpenWebText <./pretrain_corpus/README.md#openwebtext>`__

-  `General NLP Benchmarks <./general_nlp_benchmark>`__

   -  `GLUE <./general_nlp_benchmark/README.md#glue-benchmark>`__
   -  `SuperGLUE <./general_nlp_benchmark/README.md#superglue-benchmark>`__
   -  `Text Classification
      Benchmark <./general_nlp_benchmark/README.md#text-classification-benchmark>`__

Contribution Guide
------------------

We are very happy to receive and merge your contributions about new
datasets :smiley:.

To add a new dataset, you may create a ``prepare_{DATASET_NAME}.py``
file in the specific folder. Also, remember to add the documentation in
the ``README.md`` about 1) the directory structure and 2) how to use the
CLI tool for downloading + preprocessing. In addition, add citations in
the ``prepare_{DATASET_NAME}.py`` to assign credit to the original
author. Refer to the existing scripts or ask questions in Github if you
need help.

All URLs are bound with SHA1-hash keys to make sure that the downloaded
files are not corrupted. You can refer to the files in
`url\_checksums <./url_checksums>`__ for examples.

In order to generate the hash values of the data files, you can revise
`update\_download\_stats.py <update_download_stats.py>`__ and include
the new URLS + create the stats file that will store the hash keys. Use
the following command to update the hash key:

.. code:: bash

    python3 update_download_stats.py

Frequently Asked Questions
--------------------------

-  After installing GluonNLP, I cannot access the command line toolkits.
   It reports ``nlp_data: command not found``.

| The reason is that you have installed glunonnlp to a folder that is
  not in ``PATH``, e.g.,
| ``~/.local/bin``. You can try to change the ``PATH`` variable to also
  include '~/.local/bin' via the following command:

``export PATH=${PATH}:~/.local/bin``
