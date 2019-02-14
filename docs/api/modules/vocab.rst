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


ELMo Character-level Vocabulary
-------------------------------

In the original ELMo pre-trained models, the character-level vocabulary relies on UTF-8 encoding in a specific setting.
We provide the following vocabulary class to keep consistent with ELMo pre-trained models.

.. currentmodule:: gluonnlp.vocab
.. autosummary::
    :nosignatures:

    ELMoCharVocab


BERT Vocabulary
----------------

The vocabulary for BERT, inherited from :class:`gluon.Vocab` , provides some additional special tokens for ease of use.

.. currentmodule:: gluonnlp.vocab
.. autosummary::
    :nosignatures:

    BERTVocab


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
