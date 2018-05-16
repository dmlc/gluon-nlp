Vocabulary and Embedding
------------------------

All the code demonstrated in this document assumes that the following
modules or packages are imported.

.. code:: python

    >>> from mxnet import gluon, nd
    >>> import gluonnlp as nlp


Indexing words and using pre-trained word embeddings in ``gluon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a common use case, let us index words, attach pre-trained word
embeddings for them, and use such embeddings in ``gluon`` in just a few
lines of code.

To begin with, suppose that we have a simple text data set in the string
format. We can count word frequency in the data set.

.. code:: python

    >>> text_data = ['hello', 'world', 'hello', 'nice', 'world', 'hi', 'world']
    >>> counter = nlp.data.count_tokens(text_data)

The obtained ``counter`` has key-value pairs whose keys are words and
values are word frequencies. This allows us to filter out infrequent
words. Suppose that we want to build indices for all the keys in ``counter``.
We need a Vocab instance with ``counter`` as its argument.

.. code:: python

    >>> my_vocab = nlp.Vocab(counter)

To attach word embeddings to indexed words in ``my_vocab``, let us go on
to create a fastText word embedding instance by specifying the embedding
name ``fasttext`` and the pre-trained file name ``wiki.simple``.

.. code:: python

    >>> fasttext = nlp.embedding.create('fasttext', source='wiki.simple')

This automatically downloads the corresponding embedding file from public repo,
and the file is by default stored in ~/.mxnet/embedding/.
Next, we can attach word embedding ``fasttext`` to indexed words
``my_vocab``.

.. code:: python

    >>> my_vocab.set_embedding(fasttext)

Now we are ready to access the fastText word embedding vectors for
indexed words, such as 'hello' and 'world'.

.. code:: python

    >>> my_vocab.embedding[['hello', 'world']]

    [[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
        ...
       -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
     [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
        ...
       -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
    <NDArray 2x300 @cpu(0)>

To demonstrate how to use pre-trained word embeddings in the ``gluon``
package, let us first obtain indices of the words ‘hello’ and ‘world’.

.. code:: python

    >>> my_vocab[['hello', 'world']]
    [5, 4]

We can obtain the vector representation for the words ‘hello’ and
‘world’ by specifying their indices (5 and 4) and the weight matrix
``my_vocab.embedding.idx_to_vec`` in ``mxnet.gluon.nn.Embedding``.

.. code:: python

    >>> input_dim, output_dim = my_vocab.embedding.idx_to_vec.shape
    >>> layer = gluon.nn.Embedding(input_dim, output_dim)
    >>> layer.initialize()
    >>> layer.weight.set_data(my_vocab.embedding.idx_to_vec)
    >>> layer(nd.array([5, 4]))

    [[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
        ...
       -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
     [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
        ...
       -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
    <NDArray 2x300 @cpu(0)>
