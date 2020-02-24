gluonnlp.data
=============

GluonNLP Toolkit provides tools for building efficient data pipelines for NLP tasks.

.. currentmodule:: gluonnlp.data

Public Datasets
---------------

Popular datasets for NLP tasks are provided in gluonnlp.
By default, all built-in datasets are automatically downloaded from public repo and
reside in ~/.mxnet/datasets/.


Language modeling
~~~~~~~~~~~~~~~~~

`WikiText <https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/>`_
is a popular language modeling dataset from Salesforce.
It is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.
The dataset is available under the Creative Commons Attribution-ShareAlike License.

`Google 1 Billion Words <https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark/>`_
is a popular language modeling dataset.
It is a collection of over 0.8 billion tokens extracted from the WMT11 website.
The dataset is available under Apache License.

.. autosummary::
    :nosignatures:

    WikiText2
    WikiText103
    WikiText2Raw
    WikiText103Raw
    GBWStream


Text Classification
~~~~~~~~~~~~~~~~~~~

`IMDB <http://ai.stanford.edu/~amaas/data/sentiment/>`_ is a popular dataset for binary sentiment classification.
It provides a set of 25,000 highly polar movie reviews for training, 25,000 for testing, and additional unlabeled data.

`MR <https://www.cs.cornell.edu/people/pabo/movie-review-data/>`_ is a movie-review data set of 10,662 sentences labeled with respect to their overall sentiment polarity (positive or negative).

`SST-1 <http://nlp.stanford.edu/sentiment/>`_ is an extension of the MR data set. However, training/test splits are provided and labels are fine-grained (very positive, positive, neutral, negative, very negative). The training and test data sets have 237,107 and 2,210 sentences respectively.

SST-2 is the same as SST-1 with neutral sentences removed and only binary sentiment polarity are considered: very positive is considered as positive, and very negative is considered as negative.

`SUBJ <https://www.cs.cornell.edu/people/pabo/movie-review-data/>`_ is a Subjectivity data set for sentiment analysis. Sentences labeled with respect to their subjectivity status (subjective or objective).

`TREC <http://cogcomp.org/page/resource_view/49/>`_ is a movie-review data set of 10,000 sentences labeled with respect to their subjectivity status (subjective or objective).

CR is customer reviews of various products (cameras, MP3s etc.). Sentences are labeled with respect to their overall sentiment polarities (positive or negative).

`MPQA <http://www.cs.pitt.edu/mpqa/>`_ is an opinion polarity detection subtask. Sentences are labeled with respect to their overall sentiment polarities (positive or negative).

.. autosummary::
    :nosignatures:

    IMDB
    MR
    SST_1
    SST_2
    SUBJ
    TREC
    CR
    MPQA


Word Embedding Evaluation Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a number of commonly used datasets for intrinsic evaluation for word embeddings.

The similarity-based evaluation datasets include:

.. autosummary::
    :nosignatures:

    WordSim353
    MEN
    RadinskyMTurk
    RareWords
    SimLex999
    SimVerb3500
    SemEval17Task2
    BakerVerb143
    YangPowersVerb130

Analogy-based evaluation datasets include:

.. autosummary::
    :nosignatures:

    GoogleAnalogyTestSet
    BiggerAnalogyTestSet


CoNLL Datasets
~~~~~~~~~~~~~~
The `CoNLL <http://www.conll.org/previous-tasks>`_ datasets are from a series of annual
competitions held at the top tier conference of the same name. The conference is organized by SIGNLL.

These datasets include data for the shared tasks, such as part-of-speech (POS) tagging, chunking,
named entity recognition (NER), semantic role labeling (SRL), etc.

We provide built in support for CoNLL 2000 -- 2002, 2004, as well as the Universal Dependencies
dataset which is used in the 2017 and 2018 competitions.

.. autosummary::
    :nosignatures:

    CoNLL2000
    CoNLL2001
    CoNLL2002
    CoNLL2004
    UniversalDependencies21


Machine Translation Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    IWSLT2015
    WMT2014
    WMT2014BPE
    WMT2016
    WMT2016BPE


Intent Classification and Slot Labeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    ATISDataset
    SNIPSDataset


Question Answering
~~~~~~~~~~~~~~~~~~

`Stanford Question Answering Dataset (SQuAD) <https://rajpurkar.github.io/SQuAD-explorer/>`_ is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

.. autosummary::
    :nosignatures:

    SQuAD


GLUE Benchmark
~~~~~~~~~~~~~~

The `General Language Understanding Evaluation (GLUE) benchmark <https://gluebenchmark.com/>`_ is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

.. autosummary::
    :nosignatures:

    GlueCoLA
    GlueSST2
    GlueSTSB
    GlueQQP
    GlueRTE
    GlueMNLI
    GlueQNLI
    GlueWNLI
    GlueMRPC


SuperGLUE Benchmark
~~~~~~~~~~~~~~~~~~~~

The `SuperGLUE Benchmark <https://super.gluebenchmark.com>`_ a new benchmark styled after GLUE with a new set of more difficult language understanding tasks.

.. autosummary::
    :nosignatures:

    SuperGlueRTE
    SuperGlueCB
    SuperGlueWSC
    SuperGlueWiC
    SuperGlueCOPA
    SuperGlueMultiRC
    SuperGlueBoolQ
    SuperGlueReCoRD
    SuperGlueAXb
    SuperGlueAXg


Datasets
--------

Dataset API for processing common text formats. The following classes can be used or subclassed to
load custom datasets.

.. autosummary::
    :nosignatures:

    TextLineDataset
    CorpusDataset
    TSVDataset


DataStreams
-----------

DataStream API for streaming and processing common text formats. The following classes can be used or subclassed to
stream large custom data.

.. autosummary::
    :nosignatures:

    DataStream
    SimpleDataStream
    DatasetStream
    SimpleDatasetStream
    PrefetchingStream

Transforms
----------

Text data transformation functions. They can be used for processing text sequences in conjunction
with `Dataset.transform` method.

.. autosummary::
    :nosignatures:

    ClipSequence
    PadSequence
    SacreMosesTokenizer
    SpacyTokenizer
    SacreMosesDetokenizer
    BERTTokenizer
    BERTSentenceTransform

Samplers
--------

Samplers determine how to iterate through datasets. The below samplers and batch samplers can help
iterate through sequence data.

.. autosummary::
    :nosignatures:

    SortedSampler
    FixedBucketSampler
    SortedBucketSampler
    SplitSampler

The `FixedBucketSampler` uses following bucket scheme classes to generate bucket keys.

.. autosummary::
    :nosignatures:

    ConstWidthBucket
    LinearWidthBucket
    ExpWidthBucket

DataLoaders
-----------

DataLoaders loads data from a dataset and returns mini-batches of data

.. autosummary::
    :nosignatures:

    ShardedDataLoader
    DatasetLoader

Utilities
---------

Miscellaneous utility classes and functions for processing text and sequence data.

.. autosummary::
    :nosignatures:

    Counter
    count_tokens
    concat_sequence
    slice_sequence
    train_valid_split
    register
    create
    list_datasets

API Reference
-------------

.. automodule:: gluonnlp.data
   :members:
   :imported-members:
   :special-members: __iter__, __call__
