.. raw:: html

   <a href="http://gluon-nlp.mxnet.io/master/index.html"><p align="center"><img width="25%" src="https://github.com/dmlc/gluon-nlp/raw/be3bc8852155e935d68d397e0743715c54c3ce76/docs/_static/gluon_s2.png" /></a>
   </p>

.. raw:: html

   <h3 align="center">

GluonNLP: Your Choice of Deep Learning for NLP

.. raw:: html

   </h3>

.. raw:: html

   <a href='http://ci.mxnet.io/job/gluon-nlp/job/master/'><img src='https://img.shields.io/badge/python-3.6%2C3.7-blue.svg'></a>
   <a href='https://codecov.io/gh/dmlc/gluon-nlp'><img src='https://codecov.io/gh/dmlc/gluon-nlp/branch/master/graph/badge.svg'></a>
   <a href='http://ci.mxnet.io/job/gluonnlp-py3-master-gpu-doc/job/master/'><img src='http://ci.mxnet.io/buildStatus/icon?job=gluonnlp-py3-master-gpu-doc%2Fmaster'></a>
   <a href='https://pypi.org/project/gluonnlp/#history'><img src='https://img.shields.io/pypi/v/gluonnlp.svg'></a>

GluonNLP is a toolkit that enables easy text preprocessing, datasets
loading and neural models building to help you speed up your Natural
Language Processing (NLP) research.

- `Quick Start Guide <https://github.com/dmlc/gluon-nlp#quick-start-guide>`__
- `Resources <https://github.com/dmlc/gluon-nlp#resources>`__

News
====

- GluonNLP will be featured in KDD 2019 Alaska! Check out our tutorial: `From Shallow to Deep Language Representations: Pre-training, Fine-tuning, and Beyond <https://www.kdd.org/kdd2019/hands-on-tutorials>`__.
- Tutorial proposal for GluonNLP is accepted at `EMNLP 2019 <https://www.emnlp-ijcnlp2019.org>`__, Hong Kong.

- GluonNLP was featured in:

  - **JSALT 2019 in Montreal, 2019-6-14**! Checkout **https://jsalt19.mxnet.io**.
  - **AWS re:invent 2018 in Las Vegas, 2018-11-28**! Checkout `details <https://www.portal.reinvent.awsevents.com/connect/sessionDetail.ww?SESSION_ID=88736>`_.
  - **PyData 2018 NYC, 2018-10-18**! Checkout the `awesome talk <https://pydata.org/nyc2018/schedule/presentation/76/>`__ by Sneha Jha.
  - **KDD 2018 London, 2018-08-21, Apache MXNet Gluon tutorial**! Check out **https://kdd18.mxnet.io**.

Installation
============

Make sure you have Python 3.6 or newer and a recent version of MXNet (our CI
server runs the testsuite with Python 3.6).

You can install ``MXNet`` and ``GluonNLP`` using pip.

``GluonNLP`` is based on the most recent version of ``MXNet``.


In particular, if you want to install the most recent ``MXNet`` release:

::

    pip install --upgrade mxnet>=1.5.0

Else, if you want to install the most recent ``MXNet`` nightly build:

::

    pip install --pre --upgrade mxnet

Then, you can install ``GluonNLP``:

::

    pip install gluonnlp

Please check more `installation details <https://github.com/dmlc/gluon-nlp/blob/master/docs/install.rst>`_.

Docs 📖
=======

GluonNLP documentation is available at `our
website <http://gluon-nlp.mxnet.io/master/index.html>`__.

Community
=========

GluonNLP is a community that believes in sharing.

For questions, comments, and bug reports, `Github issues <https://github.com/dmlc/gluon-nlp/issues>`__ is the best way to reach us.

We now have a new Slack channel `here <https://apache-mxnet.slack.com/messages/CCCDM10V9>`__.
(`register <https://join.slack.com/t/apache-mxnet/shared_invite/enQtNDQyMjAxMjQzMTI3LTkzMzY3ZmRlNzNjNGQxODg0N2Y5NmExMjEwOTZlYmIwYTU2ZTY4ZjNlMmEzOWY5MGQ5N2QxYjhlZTFhZTVmYTc>`__).

How to Contribute
=================

GluonNLP community welcomes contributions from anyone!

There are lots of opportunities for you to become our `contributors <https://github.com/dmlc/gluon-nlp/blob/master/contributor.rst>`__:

- Ask or answer questions on `GitHub issues <https://github.com/dmlc/gluon-nlp/issues>`__.
- Propose ideas, or review proposed design ideas on `GitHub issues <https://github.com/dmlc/gluon-nlp/issues>`__.
- Improve the `documentation <http://gluon-nlp.mxnet.io/master/index.html>`__.
- Contribute bug reports `GitHub issues <https://github.com/dmlc/gluon-nlp/issues>`__.
- Write new `scripts <https://github.com/dmlc/gluon-nlp/tree/master/scripts>`__ to reproduce
  state-of-the-art results.
- Write new `examples <https://github.com/dmlc/gluon-nlp/tree/master/docs/examples>`__ to explain
  key ideas in NLP methods and models.
- Write new `public datasets <https://github.com/dmlc/gluon-nlp/tree/master/gluonnlp/data>`__
  (license permitting).
- Most importantly, if you have an idea of how to contribute, then do it!

For a list of open starter tasks, check `good first issues <https://github.com/dmlc/gluon-nlp/labels/good%20first%20issue>`__.

Also see our `contributing
guide <http://gluon-nlp.mxnet.io/master/how_to/contribute.html>`__ on simple how-tos,
contribution guidelines and more.

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

`Dataset Loading <http://gluon-nlp.mxnet.io/master/api/notes/data_api.html>`__
-------------------------------------------------------------------------------

Load the Wikitext-2 dataset, for example:

.. code:: python

    >>> import gluonnlp as nlp
    >>> train = nlp.data.WikiText2(segment='train')
    >>> train[0:5]
    ['=', 'Valkyria', 'Chronicles', 'III', '=']

`Vocabulary Construction <http://gluon-nlp.mxnet.io/master/api/modules/vocab.html>`__
-------------------------------------------------------------------------------------

Build vocabulary based on the above dataset, for example:

.. code:: python

    >>> vocab = nlp.Vocab(counter=nlp.data.Counter(train))
    >>> vocab
    Vocab(size=33280, unk="<unk>", reserved="['<pad>', '<bos>', '<eos>']")

`Neural Models Building <http://gluon-nlp.mxnet.io/master/api/modules/model.html>`__
------------------------------------------------------------------------------------

From the models package, apply a Standard RNN language model to the
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

`Word Embeddings Loading <http://gluon-nlp.mxnet.io/master/api/modules/embedding.html>`__
-----------------------------------------------------------------------------------------

For example, load a GloVe word embedding, one of the state-of-the-art
English word embeddings:

.. code:: python

    >>> glove = nlp.embedding.create('glove', source='glove.6B.50d')
    # Obtain vectors for 'baby' in the GloVe word embedding
    >>> type(glove['baby'])
    <class 'mxnet.ndarray.ndarray.NDArray'>
    >>> glove['baby'].shape
    (50,)
    
    
Reference Paper
===============

The bibtex entry for the `reference paper <https://arxiv.org/abs/1907.04433>`__ of GluonNLP is:

.. code::

   @article{gluoncvnlp2019,
     title={GluonCV and GluonNLP: Deep Learning in Computer Vision and Natural Language Processing},
     author={Guo, Jian and He, He and He, Tong and Lausen, Leonard and Li, Mu and Lin, Haibin and Shi, Xingjian and Wang, Chenguang and Xie, Junyuan and Zha, Sheng and Zhang, Aston and Zhang, Hang and Zhang, Zhi and Zhang, Zhongyue and Zheng, Shuai},
     journal={arXiv preprint arXiv:1907.04433},
     year={2019}
   }


New to Deep Learning or NLP?
============================

For background knowledge of deep learning or NLP, please refer to the open source book `Dive into Deep Learning <http://en.diveintodeeplearning.org/>`__.
