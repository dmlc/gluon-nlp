GluonNLP: a Deep Learning Toolkit for NLP
=========================================

GluonNLP provides implementations of the sate-of-the-art (SOTA) deep learning
models in NLP. It is designed for engineers, researchers, and
students to fast prototype products and research ideas based on these
models. This toolkit offers four main features:

1. Training scripts to reproduce SOTA results reported in research papers
2. A large number of pre-trained models
3. Carefully designed APIs that greatly reduce the implementation complexity
4. Community supports


This toolkit assume users has basic knowledges about deep learning and
NLP. Otherwise, please refer to introduction course such as `Stanford
CS224n <http://web.stanford.edu/class/cs224n/>`_.

.. note::

   This project is still at an early stage. Please expect that it will
   be updated frequently. We also welcome any contributions.

Installation
------------------

GluonNLP relies on the recent version of MXNet. The easiest way to install MXNet
is through `pip <https://pip.pypa.io/en/stable/installing/>`_. The following
command installs a nightly build CPU version of MXNet.

.. code-block:: bash

   pip install --pre mxnet

.. note::

   There are other pre-build MXNet packages that enables GPU supports and
   accelerate CPU performance, please refer to `this tutorial
   <http://gluon-crash-course.mxnet.io/mxnet_packages.html>`_ for details. Some
   training scripts are recommended to run on GPUs, if you don't have a GPU
   machine at hands, you may consider to `run on AWS
   <http://gluon-crash-course.mxnet.io/use_aws.html>`_.


Then install the GluonNLP toolkit by

.. code-block:: bash


   pip install -U --pre http://gluon-nlp-dist.s3-accelerate.dualstack.amazonaws.com/gluonnlp-0.1.0.tar.gz


A Quick Example
----------------

Here is a quick example that download and create a word embedding model and then
compare the similarity between two words. (You can click the go button on the
right bottom corner to run this example.)

.. raw:: html

   <iframe height="400px" width="100%"
   src="https://repl.it/@mli/gluon-nlp?lite=true" scrolling="no"
   frameborder="no" allowtransparency="true" allowfullscreen="true"
   sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin
   allow-scripts allow-modals"></iframe>

   <p></p>

Contents
--------

.. toctree::
   :maxdepth: 2

   examples/index
   api/index
   scripts/index
   how_to/contribute
