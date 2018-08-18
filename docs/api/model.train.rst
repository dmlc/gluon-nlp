gluonnlp.model.train
=====================

GluonNLP Toolkit supplies models with train-mode since the corresponding models have different behaviors in training
 and inference, e.g., the number and type of the outputs from the forward pass are different.

.. currentmodule:: gluonnlp.model.train

Language Modeling
-----------------

.. autosummary::
    :nosignatures:

    AWDRNN
    StandardRNN
    CacheCell
    get_cache_model
    BigRNN


Word Embeddings
---------------

.. autosummary::
    :nosignatures:

    EmbeddingModel
    SimpleEmbeddingModel
    FasttextEmbeddingModel


API Reference
-------------

.. automodule:: gluonnlp.model.train
    :members:
    :imported-members:
