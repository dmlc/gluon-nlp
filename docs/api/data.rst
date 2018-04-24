gluonnlp.data
=============

Gluon NLP Toolkit provides tools for building efficient data pipelines for NLP tasks.

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
    WMT2016BPE

Datasets
--------

Dataset API for processing common text formats. The following classes can be used or subclassed to
load custom datasets.

.. autosummary::
    :nosignatures:

    TextLineDataset
    CorpusDataset
    LanguageModelDataset

Transforms
----------

Text data transformation functions. They can be used for processing text sequences in conjunction
with `Dataset.transform` method.

.. autosummary::
    :nosignatures:

    ClipSequence
    PadSequence
    NLTKMosesTokenizer
    SpacyTokenizer

Samplers
--------

Samplers determine how to iterate through datasets. The below samplers and batch samplers can help
iterate through sequence data.

.. autosummary::
    :nosignatures:

    SortedSampler
    FixedBucketSampler
    SortedBucketSampler

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

.. autoclass:: gluonnlp.data.WikiText2

.. autoclass:: gluonnlp.data.WikiText103

.. autoclass:: gluonnlp.data.IMDB

.. autoclass:: gluonnlp.data.WordSim353
    :members:

.. autoclass:: gluonnlp.data.MEN
    :members:

.. autoclass:: gluonnlp.data.RadinskyMTurk
    :members:

.. autoclass:: gluonnlp.data.RareWords
    :members:

.. autoclass:: gluonnlp.data.SimLex999
    :members:

.. autoclass:: gluonnlp.data.SimVerb3500
    :members:

.. autoclass:: gluonnlp.data.SemEval17Task2
    :members:

.. autoclass:: gluonnlp.data.BakerVerb143
    :members:

.. autoclass:: gluonnlp.data.YangPowersVerb130
    :members:

.. autoclass:: gluonnlp.data.GoogleAnalogyTestSet
    :members:

.. autoclass:: gluonnlp.data.BiggerAnalogyTestSet
    :members:

.. autoclass:: gluonnlp.data.CoNLL2000

.. autoclass:: gluonnlp.data.CoNLL2001

.. autoclass:: gluonnlp.data.CoNLL2002

.. autoclass:: gluonnlp.data.CoNLL2004

.. autoclass:: gluonnlp.data.UniversalDependencies21

.. autoclass:: gluonnlp.data.IWSLT2015

.. autoclass:: gluonnlp.data.WMT2016BPE

.. autoclass:: gluonnlp.data.TextLineDataset

.. autoclass:: gluonnlp.data.CorpusDataset

.. autoclass:: gluonnlp.data.LanguageModelDataset

.. autoclass:: gluonnlp.data.ClipSequence

.. autoclass:: gluonnlp.data.PadSequence

.. autoclass:: gluonnlp.data.NLTKMosesTokenizer

.. autoclass:: gluonnlp.data.SpacyTokenizer

.. autoclass:: gluonnlp.data.SortedSampler

.. autoclass:: gluonnlp.data.FixedBucketSampler

.. autoclass:: gluonnlp.data.SortedBucketSampler

.. autoclass:: gluonnlp.data.Counter

.. autofunction:: gluonnlp.data.count_tokens

.. autofunction:: gluonnlp.data.concat_sequence

.. autofunction:: gluonnlp.data.slice_sequence

.. autofunction:: gluonnlp.data.train_valid_split

