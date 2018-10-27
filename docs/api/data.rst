gluonnlp.data
=============

GluonNLP Toolkit provides tools for building efficient data pipelines for NLP tasks.

.. currentmodule:: gluonnlp.data

Public Datasets
---------------
Popular datasets for NLP tasks are provided in gluonnlp.
By default, all built-in datasets are automatically downloaded from public repo and
reside in ~/.mxnet/datasets/.

Language modeling: WikiText
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`WikiText <https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/>`_
is a popular language modeling dataset from Salesforce.
It is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.
The dataset is available under the Creative Commons Attribution-ShareAlike License.

.. autosummary::
    :nosignatures:

    WikiText2
    WikiText103

Language modeling: Google 1 Billion Words
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Google 1 Billion Words <https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark/>`_
is a popular language modeling dataset.
It is a collection of over 0.8 billion tokens extracted from the WMT11 website.
The dataset is available under Apache License.

.. autosummary::
    :nosignatures:

    GBWStream

Sentiment Analysis: IMDB
~~~~~~~~~~~~~~~~~~~~~~~~
`IMDB <http://ai.stanford.edu/~amaas/data/sentiment/>`_ is a popular dataset for binary sentiment classification.
It provides a set of 25,000 highly polar movie reviews for training, 25,000 for testing, and additional unlabeled data.

.. autosummary::
    :nosignatures:

    IMDB

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
We provide several standard datasets for machine translation.

.. autosummary::
    :nosignatures:

    IWSLT2015
    WMT2014
    WMT2014BPE
    WMT2016
    WMT2016BPE
    SQuAD

Datasets
--------

Dataset API for processing common text formats. The following classes can be used or subclassed to
load custom datasets.

.. autosummary::
    :nosignatures:

    TextLineDataset
    CorpusDataset

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
    NLTKMosesTokenizer
    SacreMosesTokenizer
    SpacyTokenizer
    SacreMosesDetokenizer
    NLTKMosesDetokenizer

Samplers
--------

Samplers determine how to iterate through datasets. The below samplers and batch samplers can help
iterate through sequence data.

.. autosummary::
    :nosignatures:

    SortedSampler
    FixedBucketSampler
    SortedBucketSampler

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
   :special-members: __iter__
