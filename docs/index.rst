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

.. note::

   This project is at an early stage and is under active development.
   Please expect that it will be updated frequently. Contributions are welcome.

Installation
------------------

GluonNLP relies on the recent version of MXNet. The easiest way to install MXNet
is through `pip <https://pip.pypa.io/en/stable/installing/>`_. The following
command installs a nightly built CPU version of MXNet.

.. code-block:: console

   pip install --pre --upgrade mxnet

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
----------------

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


..
    .. code-block:: python

..
       import mxnet as mx
       import gluonnlp as nlp

..
       # Create a GloVe word embedding.
       glove = nlp.embedding.create('glove', source='glove.6B.50d')
       # Obtain vectors for 'baby' and 'infant' in the GloVe word embedding. 
       baby, infant = glove['baby'], glove['infant']

..
       def cos_similarity(vec1, vec2):
           # Normalize the dot product of two vectors with the L2-norm product.
           return mx.nd.dot(vec1, vec2) / (vec1.norm() * vec2.norm())

..
       print(cos_similarity(baby, infant))



Contents
--------

.. toctree::
   :glob:
   :maxdepth: 1

   self
   api/index
   examples/index
   scripts/index
   how_to/contribute
