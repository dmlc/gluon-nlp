GluonNLP: NLP made easy
=======================

Get Started: A Quick Example
----------------------------

Here is a quick example that downloads and creates a word embedding model and then
computes the cosine similarity between two words.

(You can click the play button below to run this example.)

.. container:: demo
   :name: frontpage-demo

   `Word Embedding <https://repl.it/@szha/gluon-nlp>`_

.. raw:: html

   <script type="text/javascript">
   window.onload = function() {
     var demo = document.createElement("IFRAME");
     demo.src = "https://repl.it/@szha/gluon-nlp?lite=true";
     demo.height = "400px";
     demo.width = "100%";
     demo.scrolling = "no";
     demo.frameborder = "no";
     demo.allowtransparency = true;
     demo.allowfullscreen = true;
     demo.seamless = true;
     demo.sandbox = "allow-forms allow-pointer-lock allow-same-origin allow-scripts allow-modals";
     demo_div = document.getElementById("frontpage-demo");
     while (demo_div.firstChild) {
       demo_div.removeChild(demo_div.firstChild);
     }
     demo_div.appendChild(demo);
   }; // load demo last
   </script>


.. include:: model_zoo.rst

And more in :doc:`tutorials <examples/index>`.


Installation
------------

GluonNLP relies on the recent version of MXNet. The easiest way to install MXNet
is through `pip <https://pip.pypa.io/en/stable/installing/>`_. The following
command installs the latest version of MXNet.

.. code-block:: console

   pip install --upgrade mxnet>=1.5.0

.. note::

   There are other pre-build MXNet packages that enable GPU supports and
   accelerate CPU performance, please refer to `this page
   <http://beta.mxnet.io/install.html>`_ for details. Some
   training scripts are recommended to run on GPUs, if you don't have a GPU
   machine at hand, you may consider `running on AWS
   <http://d2l.ai/chapter_appendix/aws.html>`_.


After installing MXNet, you can install the GluonNLP toolkit by

.. code-block:: console

   pip install gluonnlp


.. hint::

   For more detailed guide on installing pre-release from latest master branch,
   install from local copy of GluonNLP source code, etc.,
   click the :doc:`install <install>` link in the top navigation bar.


About GluonNLP
--------------

.. hint::

   You can find our the doc for our master development branch `here <http://gluon-nlp.mxnet.io/master/index.html>`_.

GluonNLP provides implementations of the state-of-the-art (SOTA) deep learning
models in NLP, and build blocks for text data pipelines and models.
It is designed for engineers, researchers, and students to fast prototype
research ideas and products based on these models. This toolkit offers five main features:

1. Training scripts to reproduce SOTA results reported in research papers.
2. Pre-trained models for common NLP tasks.
3. Carefully designed APIs that greatly reduce the implementation complexity.
4. Tutorials to help get started on new NLP tasks.
5. Community support.

This toolkit assumes that users have basic knowledge about deep learning and
NLP. Otherwise, please refer to an introductory course such as
`Dive into Deep Learning <https://www.d2l.ai/>`_ or
`Stanford CS224n <http://web.stanford.edu/class/cs224n/>`_.
If you are not familiar with Gluon, check out the `Gluon documentation
<http://mxnet.apache.org/versions/master/tutorials/index.html#python-tutorials>`__.
You may find the 60-min Gluon crash course linked from there especially helpful.


.. toctree::
   :hidden:
   :maxdepth: 2

   model_zoo/index
   examples/index
   api/index
   community/index
   genindex
