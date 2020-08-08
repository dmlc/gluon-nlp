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
     :opt:`Windows`

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
     :act:`Native`
     :opt:`CUDA`
     :opt:`MKL-DNN`
     :opt:`CUDA + MKL-DNN`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="native">Build-in backend for CPU.</div>
        <div class="mdl-tooltip" data-mdl-for="cuda">Required to run on Nvidia GPUs.</div>
        <div class="mdl-tooltip" data-mdl-for="mkl-dnn">Accelerate Intel CPU performance.</div>
        <div class="mdl-tooltip" data-mdl-for="cuda-mkl-dnn">Enable both Nvidia GPUs and Intel CPU acceleration.</div>

  .. admonition:: Prerequisites:

     - Requires `pip >= 9. <https://pip.pypa.io/en/stable/installing/>`_.
       Python 3.5+ are supported.

     .. container:: nightly

        - Nightly build provides latest features for enthusiasts.

  .. admonition:: Command:

     .. container:: stable

        .. container:: native

           .. code-block:: bash

              pip install --upgrade mxnet gluonnlp

        .. container:: cuda

           .. code-block:: bash

              # Here we assume CUDA 10.0 is installed. You can change the number
              # according to your own CUDA version.
              pip install --upgrade mxnet-cu100 gluonnlp

        .. container:: mkl-dnn

           .. code-block:: bash

              pip install --upgrade mxnet-mkl gluonnlp

        .. container:: cuda-mkl-dnn

           .. code-block:: bash

              # Here we assume CUDA 10.0 is installed. You can change the number
              # according to your own CUDA version.
              pip install --upgrade mxnet-cu100mkl gluonnlp

     .. container:: nightly

        .. container:: native

           .. code-block:: bash

              pip install --pre --upgrade mxnet https://github.com/dmlc/gluon-nlp/tarball/master

        .. container:: cuda

           .. code-block:: bash

              pip install --pre --upgrade mxnet-cu100 https://github.com/dmlc/gluon-nlp/tarball/master

        .. container:: mkl-dnn

           .. code-block:: bash

              pip install --pre --upgrade mxnet-mkl https://github.com/dmlc/gluon-nlp/tarball/master

        .. container:: cuda-mkl-dnn

           .. code-block:: bash

               pip install --pre --upgrade mxnet-cu100mkl https://github.com/dmlc/gluon-nlp/tarball/master

     .. container:: source

        .. container:: native

           .. code-block:: bash

              pip install --pre --upgrade mxnet
              git clone https://github.com/dmlc/gluon-nlp --branch master
              cd gluon-nlp && python setup.py install --user

        .. container:: cuda

           .. code-block:: bash

              pip install --pre --upgrade mxnet-cu100
              git clone https://github.com/dmlc/gluon-nlp
              cd gluon-nlp && python setup.py install --user

        .. container:: mkl-dnn

           .. code-block:: bash

              pip install --pre --upgrade mxnet-mkl
              git clone https://github.com/dmlc/gluon-nlp
              cd gluon-nlp && python setup.py install --user

        .. container:: cuda-mkl-dnn

           .. code-block:: bash

               pip install --pre --upgrade mxnet-cu100mkl
               git clone https://github.com/dmlc/gluon-nlp
               cd gluon-nlp && python setup.py install --user
