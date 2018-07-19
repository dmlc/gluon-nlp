.. raw:: html

   <a href="http://gluon-nlp.mxnet.io/master/index.html"><p align="center"><img width="25%" src="docs/_static/gluon_s2.png" /></a>
   </p>

.. raw:: html

   <h3 align="center">

GluonNLP: Your Choice of Deep Learning for NLP

.. raw:: html

   </h3>

.. raw:: html

   <a href='http://ci.mxnet.io/job/gluon-nlp/job/master/'><img src='https://img.shields.io/badge/python-2.7%2C%203.6-blue.svg'></a>
   <a href='https://codecov.io/gh/dmlc/gluon-nlp'><img src='https://codecov.io/gh/dmlc/gluon-nlp/branch/master/graph/badge.svg'></a>
   <a href='http://ci.mxnet.io/job/gluon-nlp/job/master/'><img src='http://ci.mxnet.io/job/gluon-nlp/job/master/badge/icon'></a>
   <a href='https://pypi.org/project/gluonnlp/#history'><img src='https://img.shields.io/pypi/v/gluonnlp.svg'></a>

GluonNLP is a toolkit that enables easy text preprocessing, datasets
loading and neural models building to help you speed up your Natural
Language Processing (NLP) research.

- `Quick Start Guide <#quick-start-guide>`__
- `Resources <#resources>`__

Installation
============

Make sure you have Python 2.7 or Python 3.6 and recent version of MXNet.
You can install ``MXNet`` and ``GluonNLP`` using pip:

::

    pip install --pre --upgrade mxnet
    pip install gluonnlp

Docs 📖
=======

GluonNLP documentation is available at `our
website <http://gluon-nlp.mxnet.io/master/index.html>`__.

Community
=========

For questions and comments, please visit our `forum <https://discuss.mxnet.io/>`__
(and `Chinese version <https://discuss.gluon.ai/>`__). For bug reports, please submit Github issues.

How to Contribute
=================

GluonNLP has been developed by community members. Everyone is
more than welcome to contribute. We together can make the GluonNLP better
and more user-friendly to more users.

Read our `contributing
guide <http://gluon-nlp.mxnet.io/master/how_to/contribute.html>`__ to get
to know about our development procedure, how to propose bug fixes and
improvements, as well as how to build and test your changes to GluonNLP.

Join our `contributors <contributor.rst>`__.

Resources
=========

Check out how to use GluonNLP for your own research or projects.

If you are new to Gluon, please check out our `60-minute crash course
<http://gluon-crash-course.mxnet.io/>`__.

For getting started quickly, refer to notebook runnable examples at
`Examples. <http://gluon-nlp.mxnet.io/master/examples/index.html>`__

For advanced examples, check out our
`Scripts. <http://gluon-nlp.mxnet.io/master/scripts/index.html>`__

For experienced users, check out our
`API Notes <http://gluon-nlp.mxnet.io/master/api/index.html>`__.

Quick Start Guide
=================

`Dataset Loading <http://gluon-nlp.mxnet.io/master/api/data.html>`__
-------------------------------------------------------------------------------------

Load the Wikitext-2 dataset, for example:

.. code:: python

    >>> import gluonnlp as nlp
    >>> train = nlp.data.WikiText2(segment='train')
    >>> train[0][0:5]
    ['=', 'Valkyria', 'Chronicles', 'III', '=']

`Vocabulary Construction <http://gluon-nlp.mxnet.io/master/api/vocab.html>`__
---------------------------------------------------------------------------------

Build vocabulary based on the above dataset, for example:

.. code:: python

    >>> vocab = nlp.Vocab(counter=nlp.data.Counter(train[0]))
    >>> vocab
    Vocab(size=33280, unk="<unk>", reserved="['<pad>', '<bos>', '<eos>']")

`Neural Models Building <http://gluon-nlp.mxnet.io/master/api/model.html>`__
-----------------------------------------------------------------------------------

From the models package, apply an Standard RNN langauge model to the
above dataset:

.. code:: python

    >>> model = nlp.model.language_model.StandardRNN('lstm', len(vocab),
    ...                                              200, 200, 2, 0.5, True)
    >>> model
    StandardRNN(
      (embedding): HybridSequential(
        (0): Embedding(33280 -> 200, float32)
        (1): Dropout(p = 0.5, axes=())
      )
      (encoder): LSTM(200 -> 200.0, TNC, num_layers=2, dropout=0.5)
      (decoder): HybridSequential(
        (0): Dense(200 -> 33280, linear)
      )
    )

`Word Embeddings Loading <http://gluon-nlp.mxnet.io/master/api/embedding.html>`__
---------------------------------------------------------------------------------

For example, load a GloVe word embedding, one of the state-of-the-art
English word embeddings:

.. code:: python

    >>> glove = nlp.embedding.create('glove', source='glove.6B.50d')
    # Obtain vectors for 'baby' in the GloVe word embedding
    >>> type(glove['baby'])
    <class 'mxnet.ndarray.ndarray.NDArray'>
    >>> glove['baby'].shape
    (50,)
