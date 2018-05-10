gluonnlp.embedding
==================

This page describes ``gluonnlp`` APIs for text embedding, such as loading pre-trained
embedding vectors for text tokens and storing them in the ``mxnet.ndarray.NDArray`` format,
and utility for intrinsic evaluation of text embeddings.

.. autosummary::
    :nosignatures:

    gluonnlp.embedding

.. currentmodule:: gluonnlp.embedding
.. autosummary::
    :nosignatures:

    register
    create
    list_sources
    TokenEmbedding
    GloVe
    FastText

API Reference
-------------

.. automodule:: gluonnlp.embedding
    :members: register, create, list_sources
.. autoclass:: gluonnlp.embedding.TokenEmbedding
    :members: from_file, serialize, deserialize
.. autoclass:: gluonnlp.embedding.GloVe
.. autoclass:: gluonnlp.embedding.FastText

.. automodule:: gluonnlp.embedding.evaluation
    :members:
