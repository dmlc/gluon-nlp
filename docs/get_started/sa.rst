Sentiment Analysis
------------------------

In this tutorial, we will learn how to load and process the sentiment dataset as well as the construction of the model.
We use `IMDB <http://ai.stanford.edu/~amaas/data/sentiment/>`_
dataset as an example, where the dataset has 50,000 movie reviews, labeled as positive or negative. The dataset
is splitted into training/testing dataset, each consisting of 25,000 reviews.

Data Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us see a quick example.

.. code:: python

    >>> import mxnet as mx
    >>> from mxnet import gluon, nd
    >>> import gluonnlp as nlp

.. code:: python

    >>> train_dataset, test_dataset = [nlp.data.IMDB(root='data/imdb', segment=segment)
    >>>                                for segment in ('train', 'test')]

.. code:: python

    >>> print('#training samples={:d}, #testing samples={:d}'.format(len(train_dataset),
    >>>                                                              len(test_dataset)))

    #training samples: 25000, #testing samples: 25000

.. code:: python

    >>> print(train_dataset[0])

    ['Bromwell High is a cartoon comedy. It ran at the same time as some other programs
    about school life, such as "Teachers". My 35 years in the teaching profession lead
    me to believe that Bromwell High\'s satire is much closer to reality than is "Teachers".
    The scramble to survive financially, the insightful students who can see right through
    their pathetic teachers\' pomp, the pettiness of the whole situation, all remind me of
    the schools I knew and their students. When I saw the episode in which a student repeatedly
    tried to burn down the school, I immediately recalled ......... at .......... High. A
    classic line: INSPECTOR: I\'m here to sack one of your teachers. STUDENT: Welcome to
    Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched.
    What a pity that it isn\'t!', 9]

In the above example, we load ``train_dataset`` and ``test_dataset``, which are both SimpleDataset objects.

``SimpleDataset``: wrapper for lists and arrays. Each entry in the train_dataset is a [string, score] pair,
where the score falls into [1, 2, ..., 10]. Thus in the given example, 9 indicates a positive feedback on the movie.


Data Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to preprocess the data so that it can be used to train the model. The following code
shows how to tokenize the string and then clip the resultant list of tokens.

.. code:: python

    >>> tokenizer = nlp.data.SpacyTokenizer('en')
    >>> length_clip = nlp.data.ClipSequence(50)
    >>> seq, score = train_dataset[0]
    >>> print(length_clip(tokenizer(seq)))

    ['Bromwell', 'High', 'is', 'a', 'cartoon', 'comedy', '.', 'It', 'ran', 'at', 'the', 'same',
    'time', 'as', 'some', 'other', 'programs', 'about', 'school', 'life', ',', 'such', 'as',
    '"', 'Teachers', '"', '.', 'My', '35', 'years', 'in', 'the', 'teaching', 'profession', 'lead',
    'me', 'to', 'believe', 'that', 'Bromwell', 'High', "'s", 'satire', 'is', 'much', 'closer',
    'to', 'reality', 'than', 'is']

Now, we are ready to preprocess the whole dataset. The following code shows how to tokenize the dataset parallelly.

.. code:: python

    >>> import time
    >>> import multiprocessing as mp

.. code:: python

    >>># Dataset preprocessing
    >>> def preprocess(x):
    >>>     data, label = x
    >>>     # In the labeled train/test sets, a negative review has a score <= 4
    >>>     # out of 10, and a positive review has a score >= 7 out of 10. Thus
    >>>     # reviews with more neutral ratings are not included in the train/test
    >>>     # sets. We labeled a negative review whose score <= 4 as 0, and a
    >>>     # positive review whose score >= 7 as 1. As the neural ratings are not
    >>>     # included in the datasets, we can simply use 5 as our threshold.
    >>>     label = int(label > 5)
    >>>     data = length_clip(tokenizer(data))
    >>>     return data, label
    >>>
    >>> def get_length(x):
    >>>     return float(len(x[0]))
    >>>
    >>> def preprocess_dataset(dataset):
    >>>     start = time.time()
    >>>     pool = mp.Pool()
    >>>     dataset = gluon.data.SimpleDataset(pool.map(preprocess, dataset))
    >>>     lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    >>>     end = time.time()
    >>>     print('Done! Tokenizing Time={:.2f}s, #Sentences={}'.format(end - start, len(dataset)))
    >>>     return dataset, lengths
    >>>
    >>> # Preprocess the dataset
    >>> train_dataset, train_data_lengths = preprocess_dataset(train_dataset)
    >>> test_dataset, test_data_lengths = preprocess_dataset(test_dataset)

    Tokenize using spaCy...

    Done! Tokenizing Time=11.68s, #Sentences=25000

    Done! Tokenizing Time=11.65s, #Sentences=25000

Then, we are going to construct a vocabulary for the training dataset. The vocabulary will be used
to convert the tokens to numerical indices, which facilitates the creation of word embedding matrices.

.. code:: python

    >>> import itertools
    >>> train_seqs = [sample[0] for sample in train_dataset]
    >>> counter = nlp.data.count_tokens(list(itertools.chain.from_iterable(train_seqs)))
    >>> vocab = nlp.Vocab(counter, max_size=10000, padding_token=None,
    >>>                   bos_token=None, eos_token=None)
    >>> print(vocab)

    Vocab(size=10001, unk="<unk>", reserved="None")

.. code:: python

    >>># Convert string token to its index in the dictionary
    >>> def token_to_idx(x):
    >>>     return vocab[x[0]], x[1]
    >>>
    >>> pool = mp.Pool()
    >>> train_dataset = pool.map(token_to_idx, train_dataset)
    >>> test_dataset = pool.map(token_to_idx, test_dataset)
    >>> pool.close()
    >>> print(train_dataset[0][0])

    [0, 1456, 9, 4, 854, 174, 3, 37, 2081, 40, 1, 206, 61, 23, 59, 115, 6287, 39, 382, 139, 2, 169,
    23, 15, 0, 15, 3, 241, 3665, 119, 12, 1, 6785, 7237, 562, 68, 8, 249, 16, 0, 1456, 17, 1673, 9,
    99, 4149, 8, 828, 100, 9]

Create the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then, we are going to create the model for the sentiment analysis. Our model is composed of a
two-layer LSTM followed by an average pooling and a sigmoid output layer as illustrated in the
Figure above. From the embedding layer, the new representations will be passed to LSTM cells.
These will include information about the sequence of words in the data. Thus, given an input
sequence, the memory cells in the LSTM layer will produce a representation sequence. This
representation sequence is then averaged over all timesteps resulting in representation h.
Finally, this representation is fed to a sigmoid output layer. We’re using the sigmoid because
we’re trying to predict if this text has positive or negative sentiment, and a sigmoid activation
function allows the model to compute the posterior probability.

.. code:: python

    >>> from nlp.model.utils import _get_rnn_layer
    >>> class SentimentNet(gluon.Block):
    >>>    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0,
    >>>                 prefix=None, params=None):
    >>>        super(SentimentNet, self).__init__(prefix=prefix, params=params)
    >>>        with self.name_scope():
    >>>            self.embedding =  nn.Embedding(vocab_size, embed_size)
    >>>            self.encoder = _get_rnn_layer('rnn_relu', 2, embed_size, hidden_size, dropout, 0)
    >>>            self.out_layer = gluon.nn.HybridSequential()
    >>>            with self.out_layer.name_scope():
    >>>                self.out_layer.add(gluon.nn.Dense(1, flatten=False))
    >>>
    >>>    def forward(self, data, valid_length):
    >>>        # Shape(T, N, C)
    >>>        encoded = self.encoder(nd.Dropout(self.embedding(data), 0.2, axes=(0,)))
    >>>        # Zero out the values with position exceeding the valid length
    >>>        masked_encoded = nd.SequenceMask(encoded,
    >>>                                         sequence_length=valid_length,
    >>>                                         use_sequence_length=True)
    >>>        agg_state = nd.broadcast_div(nd.sum(masked_encoded, axis=0),
    >>>                                     nd.expand_dims(valid_length, axis=1))
    >>>        out = self.out_layer(agg_state)
    >>>        return out
    >>>
    >>> net = SentimentNet(200, 200)
    >>> net.hybridize()
    >>> net.initialize(mx.init.Xavier(), ctx=mx.gpu(0))
    >>> print(net)


The function ``_get_rnn_layer`` returns a 2-layer LSTM with ReLU activation.
More detail about the training using pretrained language model and bucketing can be found in the following:

.. toctree::
   :maxdepth: 1

   ../examples／sentiment_analysis/sentiment_analysis.ipynb


