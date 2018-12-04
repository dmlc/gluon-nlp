Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

:download:`[Download] </model_zoo/bert.zip>`

Reference: Devlin, Jacob, et al. "`Bert: Pre-training of deep bidirectional transformers for language understanding. <https://arxiv.org/abs/1810.04805>`_" arXiv preprint arXiv:1810.04805 (2018).

The following pre-trained BERT models are available from the **gluonnlp.model.get_model** API:

+--------------------+---------------------------------+-------------------------------+--------------------+
|                    | book_corpus_wiki_en_uncased     | book_corpus_wiki_en_cased     | wiki_multilingual  |
+====================+=================================+===============================+====================+
| bert_12_768_12     | ✓                               | ✓                             | ✓                  |
+--------------------+---------------------------------+-------------------------------+--------------------+
| bert_24_1024_16    | x                               | ✓                             | x                  |
+--------------------+---------------------------------+-------------------------------+--------------------+

where **bert_12_768_12** refers to the BERT BASE model, and **bert_24_1024_16** refers to the BERT LARGE model.

BERT for Sentence Pair Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example script to fine-tune sentence pair classification with pre-trained
BERT model.

Download the MRPC dataset:

 .. code-block:: console

    $ curl -L https://tinyurl.com/yaznh3os -o download_glue_data.py
    $ python3 download_glue_data.py --data_dir glue_data --tasks MRPC

Use the following command to fine-tune the BERT model for classification on the MRPC dataset.

.. code-block:: console

   $ GLUE_DIR=glue_data MXNET_GPU_MEM_POOL_TYPE=Round python3 finetune_classifier.py --batch_size 32 --optimizer adam --epochs 3 --gpu

It gets validation accuracy of 87.0%, whereas the the original Tensorflow implementation give evaluation results between 84% and 88%.
