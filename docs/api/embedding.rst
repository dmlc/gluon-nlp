gluonnlp.embedding
==================

GluonNLP Toolkit provides tools for working with embeddings.

.. currentmodule:: gluonnlp.embedding

This page describes the ``gluonnlp`` APIs for text embedding, such as loading
pre-trained embedding vectors for text tokens and storing them in the
``mxnet.ndarray.NDArray`` format as well as utilities for intrinsic evaluation
of text embeddings.


Pre-trained Embeddings
----------------------

.. currentmodule:: gluonnlp.embedding
.. autosummary::
    :nosignatures:

    register
    create
    list_sources
    TokenEmbedding
    GloVe
    FastText


Intrinsic evaluation
--------------------

.. currentmodule:: gluonnlp.embedding.evaluation
.. autosummary::
    :nosignatures:

    register
    create
    list_evaluation_functions
    WordEmbeddingSimilarityFunction
    WordEmbeddingAnalogyFunction
    CosineSimilarity
    ThreeCosAdd
    ThreeCosMul
    WordEmbeddingSimilarity
    WordEmbeddingAnalogy


API Reference
-------------

.. automodule:: gluonnlp.embedding
    :members:
    :imported-members:
    :special-members: __contains__, __getitem__, __setitem__

.. automodule:: gluonnlp.embedding.evaluation
    :members:
    :imported-members:
    :special-members: __contains__, __getitem__, __setitem__
