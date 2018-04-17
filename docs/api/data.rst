Gluon NLP Datasets and Data API
===============================

Gluon NLP Toolkit provides tools for building efficient data pipelines for NLP tasks.

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

.. autoclass:: gluonnlp.data.WikiText2
.. autoclass:: gluonnlp.data.WikiText103

Sentiment Analysis: IMDB
~~~~~~~~~~~~~~~~~~~~~~~~
`IMDB <http://ai.stanford.edu/~amaas/data/sentiment/>`_ is a popular dataset for binary sentiment classification.
It provides a set of 25,000 highly polar movie reviews for training, 25,000 for testing, and additional unlabeled data.

.. autoclass:: gluonnlp.data.IMDB

Word Embedding Evaluation Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are a number of commonly used datasets for intrinsic evaluation for word embeddings.

The similarity-based evaluation datasets include:

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

Analogy-based evaluation datasets include:

.. autoclass:: gluonnlp.data.GoogleAnalogyTestSet
    :members:

.. autoclass:: gluonnlp.data.BiggerAnalogyTestSet
    :members:

Datasets
--------

.. automodule:: gluonnlp.data.dataset
    :members:

Transformers
------------

.. automodule:: gluonnlp.data.transforms
    :members:

Batch Loaders
-------------

.. automodule:: gluonnlp.data.batchify
    :members:

Samplers
--------

.. automodule:: gluonnlp.data.sampler
    :members:

Utilities
---------

.. automodule:: gluonnlp.data.utils
    :members:
