Vocabulary and Embedding API
============================

Overview
--------

This page describes ``gluonnlp`` APIs for text data numericalization,
such as bulding look-up indices and loading pre-trained
embedding vectors for text tokens and storing them in the
``mxnet.ndarray.NDArray`` format:

.. autoclass:: gluonnlp.Vocab

.. autosummary::
    :nosignatures:

    gluonnlp.embedding

All the code demonstrated in this document assumes that the following
modules or packages are imported.

.. code:: python

    >>> from mxnet import gluon, nd
    >>> import gluonnlp as nlp


Vocabulary
----------

The vocabulary builds indices for text tokens and can be attached with
token embeddings. The input counter whose keys are candidate indices may
be obtained via :func:`gluonnlp.data.count_tokens`

.. currentmodule:: gluonnlp.vocab
.. autosummary::
    :nosignatures:

    Vocab

Suppose that we have a simple text data set as tokenized list. We can
count word frequency in the data set.

.. code:: python

    >>> counter = nlp.data.count_tokens(text_data)

Suppose that we want to build indices for the 2 most frequent
keys in ``counter`` with the default setting, which includes the unknown token
representation ``'<unk>'`` and reserved tokens ``'<pad>', '<bos>', '<eos>'``.

.. code:: python

    >>> my_vocab = nlp.Vocab(counter, max_size=2)

We can access properties such as ``token_to_idx`` (mapping tokens to
indices), ``idx_to_token`` (mapping indices to tokens),
``unknown_token`` (representation of any unknown token) and
``reserved_tokens`` (reserved tokens).

.. code:: python

    >>> my_vocab.token_to_idx
    {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3, 'world': 4, 'hello': 5}
    >>> my_vocab.idx_to_token
    ['<unk>', '<pad>', '<bos>', '<eos>', 'world', 'hello']
    >>> my_vocab.unknown_token
    '<unk>'
    >>> my_vocab.reserved_tokens
    ['<pad>', '<bos>', '<eos>']
    >>> len(my_vocab)
    6
    >>> my_vocab[['hello', 'world']]
    [5, 4]

Besides the specified unknown token ``'<unk>'`` and reserved_token ``'<pad>', '<bos>', '<eos>'``
are indexed, the 2 most frequent words ‘world’ and ‘hello’ are also
indexed.

Attach token embedding to vocabulary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One or more token embeddings can be attached to vocabulary instance.

To begin with, suppose that we have a vocabulary from simple text data set as before.

.. code:: python

    >>> my_vocab = nlp.Vocab(counter, max_size=2)

Let us define the fastText word embedding instance with the pre-trained
file ``wiki.simple.vec``.

.. code:: python

    >>> fasttext = nlp.embedding.create('fasttext', source='wiki.simple.vec')

So we can attach word embedding ``fasttext`` to indexed words
``my_vocab``.

.. code:: python

    >>> my_vocab.set_embedding(fasttext)

Now we are ready to access the fastText word embedding vectors for the
indexed words.

.. code:: python

    >>> my_vocab.embedding[['hello', 'world']]

    [[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
        ...
       -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
     [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
        ...
       -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
    <NDArray 2x300 @cpu(0)>

We can similarly define the GloVe word embedding with the pre-trained file
``glove.6B.50d.txt``, and attach the GloVe text embedding to the vocabulary instance.

.. code:: python

    >>> glove = nlp.embedding.create('glove', source='glove.6B.50d.txt')
    >>> my_vocab.set_embedding(glove)

Now we are ready to access the GloVe word embedding vectors for the
indexed words.

.. code:: python

    >>> my_vocab.set_embedding[['hello', 'world']]

    [[  -0.38497001  0.80092001
        ...
        0.048833    0.67203999]
     [  -0.41486001  0.71847999
        ...
       -0.37639001 -0.67541999]]
    <NDArray 2x50 @cpu(0)>

If a token is unknown to ``my_vocab``, its embedding vector is
initialized according to the default specification in ``glove`` (all
elements are 0).

.. code:: python


    >>> my_vocab.embedding['nice']

    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      ...
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    <NDArray 50 @cpu(0)>

Pre-trained word embeddings
---------------------------

To load token embeddings from an externally hosted pre-trained token
embedding file, such as those of GloVe and FastText, use
:func:`gluonnlp.embedding.create`.

To get all the available ``embedding_name`` and ``source``, use
:func:`gluonnlp.embedding.list_sources`.

.. code:: python

    >>> text.embedding.list_sources()
    {'glove': ['glove.42B.300d.txt', 'glove.6B.50d.txt', 'glove.6B.100d.txt', ...],
     'fasttext': ['wiki.en.vec', 'wiki.simple.vec', 'wiki.zh.vec', ...]}

Alternatively, to load embedding vectors from a custom pre-trained text
token embedding file, use
:meth:`gluonnlp.embedding.TokenEmbedding.from_file`.

.. currentmodule:: gluonnlp.embedding
.. autosummary::
    :nosignatures:

    register
    create
    list_sources
    TokenEmbedding
    GloVe
    FastText

Implement a new text token embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a new ``embedding``, create a subclass of
``gluonnlp.embedding.TokenEmbedding``. Also add
``@gluonnlp.embedding.register`` before this
class.

Word embeddings evaluation
--------------------------
.. automodule:: gluonnlp.embedding.evaluation
    :members:

API Reference
-------------

.. raw:: html

   <script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

.. automodule:: gluonnlp.embedding
    :members: register, create, list_sources
.. autoclass:: gluonnlp.embedding.TokenEmbedding
    :members: from_file
.. autoclass:: gluonnlp.embedding.GloVe
.. autoclass:: gluonnlp.embedding.FastText

.. automodule:: gluonnlp.vocab
.. autoclass:: gluonnlp.vocab.Vocab
    :members: set_embedding, to_tokens, to_indices

.. raw:: html

   <script>auto_index("api-reference");</script>
