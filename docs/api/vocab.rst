gluonnlp
========

This page describes ``gluonnlp.Vocab``, API for text data numericalization.


Vocabulary
----------

The vocabulary builds indices for text tokens and can be attached with
token embeddings. The input counter whose keys are candidate indices may
be obtained via :func:`gluonnlp.data.count_tokens`

.. currentmodule:: gluonnlp.vocab
.. autosummary::
    :nosignatures:

    Vocab

API Reference
-------------

.. autoclass:: gluonnlp.Vocab
    :members: set_embedding, to_tokens, to_indices

.. raw:: html

   <script>auto_index("api-reference");</script>
