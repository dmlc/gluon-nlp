Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

:download:`[Download] </model_zoo/bert.zip>`

BERT for Sentence Pair Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the MRPC dataset:

 .. code-block:: console

    $ curl https://tinyurl.com/yaznh3os -o download_glue_data.py
    $ python3 download_glue_data.py --data_dir glue_data --tasks MRPC

Use the following command to fine-tune the BERT model for classification on the MRPC dataset.

.. code-block:: console

   $ GLUE_DIR=glue_data MXNET_GPU_MEM_POOL_TYPE=Round python3 finetune_classifier.py --batch_size 32 --optimizer adam --epochs 3 --gpu

It gets validation accuracy of 86%.
