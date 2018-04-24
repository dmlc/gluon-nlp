.. raw:: html

   <p align="center"><img width="30%" src="docs/_static/gluon_s2.png" />
   </p>

.. raw:: html

   <h3 align="center">

Gluon NLP: Your Choice for Deep Learning for NLP

.. raw:: html

   </h3>

.. raw:: html

   <a href='http://ci.mxnet.io/job/gluon-nlp/job/master/'><img src='https://img.shields.io/badge/python-2.7%2C%203.6-blue.svg'></a>
   <a href='http://ci.mxnet.io/job/gluon-nlp/job/master/'><img src='https://codecov.io/gh/leezu/gluon-nlp/branch/master/graph/badge.svg?token=xQ2HKDk9ux'></a>
   <a href='http://ci.mxnet.io/job/gluon-nlp/job/master/'><img src='http://ci.mxnet.io/job/gluon-nlp/job/master/badge/icon'></a>

Gluonnlp is a toolkit that enables easy text preprocessing, datasets
loading and neural models building to help you speed up your Natural
Language Processing (NLP) research.

Work with us to add your own datasets and neural network models! Please follow `contributing
guide <http://gluon-nlp.mxnet.io/master/how_to/contribute.html>`__ to contribute if
you are willing to collaborate with us.

Installation
============

Make sure you have Python 2.7 or Python 3.6 and recent version of MXNet.
You can install ``MXNet`` and ``gluonnlp`` using pip:

::

    pip install --pre --upgrade mxnet
    pip install gluonnlp

Docs 📖
======

Gluonnlp documentation is available at `our
website <http://gluon-nlp.mxnet.io/master/api/index.html>`__.

Basics
======

Check out how to use Gluonnlp for your own research or projects.

`Data <http://gluon-nlp.mxnet.io/master/api/index.html#data-processing>`__ Loading
----------------------------------------------------------------------------------

Load the Wikitext-2 dataset, for example:

.. code:: python

    >>> import os
    >>> import mxnet as mx
    >>> import gluonnlp as nlp

    # Load the Wikitext-2 training dataset
    >>>train = nlp.data.WikiText2(
    >>>        segment='train', root=os.path.join('tests', 'data', 'wikitext-2'))
    >>>train[0][0:10]
    ['=', 'Valkyria', 'Chronicles', 'III', '=', '<eos>', 'Senjō', 'no', 'Valkyria', '3']

`Vocabulary <http://gluon-nlp.mxnet.io/master/api/vocab_emb.html>`__ Construction
---------------------------------------------------------------------------------

Build vocabulary based on the above dataset, for example:

.. code:: python

    >>> vocab = nlp.Vocab(counter=nlp.data.Counter(train[0]), padding_token=None, bos_token=None)
    >>> vocab
    Vocab(size=33278, unk="<unk>", reserved="['<eos>']")

`Neural Models <http://gluon-nlp.mxnet.io/master/api/index.html#model>`__ Building
----------------------------------------------------------------------------------

From the models package, apply an Standard RNN langauge model to the
above dataset:

.. code:: python

    >>> model = nlp.model.language_model.StandardRNN('lstm', len(vocab),
    200, 200, 2, 0.5, True)
    >>> model

    StandardRNN(
      (embedding): HybridSequential(
        (0): Embedding(33278 -> 200, float32)
        (1): Dropout(p = 0.5, axes=())
      )
      (encoder): LSTM(200 -> 200.0, TNC, num_layers=2, dropout=0.5)
      (decoder): HybridSequential(
        (0): Dense(200 -> 33278, linear)
      )
    )

`Word Embeddings <http://gluon-nlp.mxnet.io/master/api/vocab_emb.html>`__ Loading
---------------------------------------------------------------------------------

For example, load a GloVe word embedding, one of the state-of-the-art
English word embeddings:

.. code:: python

    >>> glove = nlp.embedding.create('glove', source='glove.6B.50d')
    # Obtain vectors for 'baby' in the GloVe word embedding
    >>> glove['baby']
    [ 0.54936   0.22994  -0.035731 -0.91432   0.70442   1.3736   -0.99369
     -0.50342   0.5793    0.34814   0.23851   0.54439   0.34322   0.57407
      1.3732    0.46358  -0.72877   0.28868   0.10006  -0.2302   -0.12893
      0.7033    0.39612   0.26045   0.26971  -1.3036   -0.93774   0.27053
      0.60701  -0.66894   1.9709    0.6796   -0.69439   1.038     0.51364
      0.23022   0.36456  -0.30902   1.1395   -1.1466   -0.78887   0.054432
     -0.069112 -0.24386   1.4049    0.091876  0.23131  -1.3028    0.3246
      0.10741 ]
    <NDArray 50 @cpu(0)>

More Examples
-------------

For getting started quickly, refer to notebook runnable examples at
`Examples. <http://gluon-nlp.mxnet.io/master/examples/index.html>`__

If you have more questions about the usage, please refer to more advanced
examples at
`Scripts. <http://gluon-nlp.mxnet.io/master/scripts/index.html>`__

For example, we have the SOTA language model and sentiment analysis model
available by using the following scripts to train the corresponding models.

- :download:`Language model script <language_model/word_language_model.py>`.

- :download:`Sentiment analysis script <sentiment_analysis.py>`.

More Help
---------

If more help is needed, please ask your questions at our `Gluonnlp
discussion forum <https://discuss.mxnet.io/>`__. If you understand
Chinese, you can also ask at `Chinese version <https://discuss.gluon.ai/>`__.

How to Contribute
=================

Gluon NLP toolkit has been developed by community members. Everyone is
more than welcome to contribute. It is a way to make the project better
and more accessible to more users.

Contribute Now
------------------

Read our `contributing
guide <http://gluon-nlp.mxnet.io/master/how_to/contribute.html>`__ to
learn about our development process, how to propose bugfixes and
improvements, and how to build and test your changes to Gluonnlp.

Gluonnlp Maintainers (Ordered by last name alphabetical order)
-------------------------------------------------------------

-  `Lausen, Leonard<https://github.com/leezu>`__
-  `Li, Mu <https://github.com/mli>`__
-  `Shi, Xingjian <https://github.com/sxjscience>`__
-  `Wang, Chenguang <https://github.com/cgraywang>`__
-  `Zha, Sheng <https://github.com/szha>`__
-  `Zhang, Aston <https://github.com/astonzhang>`__
-  `Zheng, Shuai <https://github.com/szhengac>`__
