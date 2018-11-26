Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

:download:`[Download] </model_zoo/bert.zip>`

BERT for Sentence Pair Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the MRPC dataset:

 .. code-block:: console
    $ curl https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/becd574dd938f045ea5bd3cb77d1d506541b5345/download_glue_data.py -o download_glue_data.py
    $ python download_glue_data.py --data_dir glue_data --tasks MRPC

Use the following command to fine-tune the BERT model for classification on the MRPC dataset.

.. code-block:: console

   $ GLUE_DIR=glue_data MXNET_GPU_MEM_POOL_TYPE=Round python3 finetune_classifier.py --batch_size 32 --optimizer adam --epochs 3 --gpu

It gets validation accuracy of 86%.
