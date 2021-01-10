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


.. include:: examples.rst

And more in :doc:`tutorials <tutorials/index>`.


.. include:: install.rst


About GluonNLP
--------------

.. hint::

   You can find out the doc for our master development branch `here <http://nlp.gluon.ai/master/index.html>`_.

GluonNLP provides implementations of the state-of-the-art (SOTA) deep learning
models in NLP, and build blocks for text data pipelines and models.
It is designed for engineers, researchers, and students to fast prototype
research ideas and products based on these models. This toolkit offers five main features:

1. Carefully designed APIs that greatly reduce the implementation complexity.
2. Pre-trained models for common NLP tasks.
3. Tutorials to help get started on new NLP tasks.
4. Community support.

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

   tutorials/index
   model_zoo/index
   api/index
   website/index
   genindex
