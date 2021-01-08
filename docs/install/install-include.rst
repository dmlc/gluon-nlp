Select your preferences and run the install command.

.. role:: title
.. role:: opt
   :class: option
.. role:: act
   :class: active option

.. container:: install

  .. container:: opt-group

     :title:`OS:`
     :opt:`Linux`
     :opt:`macOS`

  .. container:: opt-group

     :title:`Version:`
     :act:`Stable`
     :opt:`Nightly`
     :opt:`Source`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="stable">Stable Release.</div>
        <div class="mdl-tooltip" data-mdl-for="nightly">Nightly build with latest features.</div>
        <div class="mdl-tooltip" data-mdl-for="source">Install GluonNLP from source.</div>


  .. container:: opt-group

     :title:`Backend:`
     :act:`CPU-Only`
     :opt:`CUDA`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="cpu-only">Build-in backend for CPU.</div>
        <div class="mdl-tooltip" data-mdl-for="cuda">Required to run on Nvidia GPUs.</div>

  .. admonition:: Prerequisites:

     - Requires `pip >= 19.0.0. <https://pip.pypa.io/en/stable/installing/>`_.
       Python 3.6+ are supported.

     .. container:: nightly

        - Nightly build provides latest features for enthusiasts.

  .. admonition:: Command:

     .. container:: stable

        .. container:: cpu-only

           .. code-block:: bash

              # Install MXNet
              python3 -m pip install -U --pre "mxnet>=2.0.0" -f https://dist.mxnet.io/python

              # Install GluonNLP
              git clone https://github.com/dmlc/gluon-nlp.git
              cd gluon-nlp
              python3 -m pip install -U -e ."[extras]"

        .. container:: cuda

           .. code-block:: bash

              # Here we assume CUDA 10.2 is installed. You can change the number
              # according to your own CUDA version, e.g., cu101, cu110
              python3 -m pip install -U --pre "mxnet-cu102>=2.0.0" -f https://dist.mxnet.io/python

              # Install GluonNLP
              git clone https://github.com/dmlc/gluon-nlp.git
              cd gluon-nlp
              python3 -m pip install -U -e ."[extras]"

     .. container:: source

        .. container:: cpu-only

           .. code-block:: bash

              # Install MXNet
              python3 -m pip install -U --pre "mxnet>=2.0.0" -f https://dist.mxnet.io/python

              # Install GluonNLP
              git clone https://github.com/dmlc/gluon-nlp.git
              cd gluon-nlp
              python3 -m pip install -U -e ."[extras]"

        .. container:: cuda

           .. code-block:: bash

              # Here we assume CUDA 10.2 is installed. You can change the number
              # according to your own CUDA version, e.g., cu100, cu101
              python3 -m pip install -U --pre "mxnet-cu102>=2.0.0" -f https://dist.mxnet.io/python

              # Install GluonNLP
              git clone https://github.com/dmlc/gluon-nlp.git
              cd gluon-nlp
              python3 -m pip install -U -e ."[extras]"
