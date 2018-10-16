GluonNLP: a Deep Learning Toolkit for Natural Language Processing (NLP)
=======================================================================

GluonNLP provides implementations of the state-of-the-art (SOTA) deep learning
models in NLP, and build blocks for text data pipelines and models.
It is designed for engineers, researchers, and students to fast prototype
research ideas and products based on these models. This toolkit offers four main features:

1. Training scripts to reproduce SOTA results reported in research papers.
2. Pre-trained models for common NLP tasks.
3. Carefully designed APIs that greatly reduce the implementation complexity.
4. Community support.


This toolkit assumes that users have basic knowledge about deep learning and
NLP. Otherwise, please refer to an introduction course such as
`Deep Learning---The Straight Dope <http://gluon.mxnet.io/>`_ or
`Stanford CS224n <http://web.stanford.edu/class/cs224n/>`_.

The toolkit supports the following NLP tasks:

1. :doc:`model_zoo/word_embeddings/index`
2. :doc:`model_zoo/language_model/index`
3. :doc:`model_zoo/machine_translation/index`
4. :doc:`model_zoo/text_classification/index`
5. :doc:`model_zoo/sentiment_analysis/index`
6. :doc:`model_zoo/text_generation/index`

.. note::

   We can be found on `Github <https://github.com/dmlc/gluon-nlp>`_. This project is at an early stage
   and is under active development, so please expect frequent updates.
   Check out our `contribution guide <http://gluon-nlp.mxnet.io/how_to/contribute.html>`_
   to see how you can help.


.. hint::

   You can find our the doc for our master development branch `here <http://gluon-nlp.mxnet.io/master/index.html>`_.


Installation
------------

GluonNLP relies on the recent version of MXNet. The easiest way to install MXNet
is through `pip <https://pip.pypa.io/en/stable/installing/>`_. The following
command installs a nightly built CPU version of MXNet.

.. code-block:: console

   pip install --upgrade mxnet==1.3.0

.. note::

   There are other pre-build MXNet packages that enable GPU supports and
   accelerate CPU performance, please refer to `this tutorial
   <http://gluon-crash-course.mxnet.io/mxnet_packages.html>`_ for details. Some
   training scripts are recommended to run on GPUs, if you don't have a GPU
   machine at hands, you may consider `running on AWS
   <http://gluon-crash-course.mxnet.io/use_aws.html>`_.


Then install the GluonNLP toolkit by

.. code-block:: console

   pip install gluonnlp


A Quick Example
---------------

Here is a quick example that downloads and creates a word embedding model and then
computes the cosine similarity between two words.

(You can click the go button on the right bottom corner to run this example.)

.. raw:: html

  <iframe height="400px" width="100%"
  src="https://repl.it/@szha/gluon-nlp?lite=true" scrolling="no"
  frameborder="no" allowtransparency="true" allowfullscreen="true"
  sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin
  allow-scripts allow-modals"></iframe>

  <p></p>


Contents
--------


.. toctree::
   :maxdepth: 1
   :caption: Model Zoo

   model_zoo/word_embeddings/index
   model_zoo/language_model/index
   model_zoo/machine_translation/index
   model_zoo/text_classification/index
   model_zoo/sentiment_analysis/index
   model_zoo/text_generation/index


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   examples/word_embedding/word_embedding.ipynb
   examples/word_embedding/word_embedding_training.ipynb
   examples/language_model/language_model.ipynb
   examples/sentiment_analysis/sentiment_analysis.ipynb
   examples/machine_translation/gnmt.ipynb
   examples/machine_translation/transformer.ipynb


.. toctree::
   :maxdepth: 1
   :caption: API

   api/index
   api/notes/data_api.rst
   api/notes/vocab_emb.rst
   api/notes/sequence_sampling.rst


.. toctree::
   :maxdepth: 1
   :caption: Community

   how_to/contribute
