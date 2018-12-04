:orphan:

Installation
~~~~~~~~~~~~

GluonNLP relies on the recent version of MXNet. The easiest way to install MXNet
is through `pip <https://pip.pypa.io/en/stable/installing/>`_. The following
command installs the latest version of MXNet.

.. code-block:: console

   pip install --upgrade mxnet>=1.3.0

.. note::

   There are other pre-build MXNet packages that enable GPU supports and
   accelerate CPU performance, please refer to `this tutorial
   <http://gluon-crash-course.mxnet.io/mxnet_packages.html>`_ for details. Some
   training scripts are recommended to run on GPUs, if you don't have a GPU
   machine at hands, you may consider `running on AWS
   <http://gluon-crash-course.mxnet.io/use_aws.html>`_.


After installing MXNet, you can install the GluonNLP toolkit by

.. code-block:: console

   pip install gluonnlp

