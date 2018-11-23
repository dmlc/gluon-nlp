gluonnlp.vocab
==============

This page describes the ``gluonnlp.Vocab`` class for text data numericalization
and the subword functionality provided in ``gluonnlp.vocab``.


Vocabulary
----------

The vocabulary builds indices for text tokens and can be attached with
token embeddings. The input counter whose keys are candidate indices may
be obtained via :func:`gluonnlp.data.count_tokens`

.. currentmodule:: gluonnlp
.. autosummary::
    :nosignatures:

    Vocab


Subword functionality
---------------------

When using a vocabulary of fixed size, out of vocabulary words may be
encountered. However, words are composed of characters, allowing intelligent
fallbacks for out of vocabulary words based on subword units such as the
characters or ngrams in a word. :class:`gluonnlp.vocab.SubwordFunction` provides
an API to map words to their subword units. :doc:`model.train` contains
models that make use of subword information to word embeddings.

.. currentmodule:: gluonnlp.vocab
.. autosummary::
    :nosignatures:

    SubwordFunction
    ByteSubwords
    NGramHashes


API Reference
-------------

.. automodule:: gluonnlp
    :members:
    :imported-members:
    :special-members: __call__, __len__

.. automodule:: gluonnlp.vocab
    :exclude-members: Vocab
    :members:
    :imported-members:
    :special-members: __call__, __len__
