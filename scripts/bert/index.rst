Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

:download:`[Download] </model_zoo/bert.zip>`

Reference: Devlin, Jacob, et al. "`Bert: Pre-training of deep bidirectional transformers for language understanding. <https://arxiv.org/abs/1810.04805>`_" arXiv preprint arXiv:1810.04805 (2018).

The following pre-trained BERT models are available from the **gluonnlp.model.get_model** API:

+--------------------+---------------------------------+-------------------------------+--------------------+-------------------------+---------+
|                    | book_corpus_wiki_en_uncased     | book_corpus_wiki_en_cased     | wiki_multilingual  | wiki_multilingual_cased | wiki_cn |
+====================+=================================+===============================+====================+=========================+=========+
| bert_12_768_12     | ✓                               | ✓                             | ✓                  | ✓                       | ✓       |
+--------------------+---------------------------------+-------------------------------+--------------------+-------------------------+---------+
| bert_24_1024_16    | ✓                               | ✓                             | x                  | x                       | x       |
+--------------------+---------------------------------+-------------------------------+--------------------+-------------------------+---------+

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

   $ GLUE_DIR=glue_data python finetune_classifier.py --batch_size 32 --optimizer bertadam --epochs 3 --gpu --seed 1 --lr 2e-5

It gets validation accuracy of 87.3%, whereas the the original Tensorflow implementation give evaluation results between 84% and 88%.

BERT for SQuAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example script to fine-tune SQuAD with pre-trained
BERT model.

Use the following command to fine-tune the BERT model for SQuAD1.1 dataset.

Note that this will require more than 12G of GPU memory.
 
.. code-block:: console

    $ python finetune_squad.py --optimizer adam --gpu

If you are using less than 12G of GPU memory, you can use the following command to achieve a similar effect. But need Mxnet>1.5.0

Note that this will require approximately no more than 8G of GPU memory. If your GPU memory is too small, you can adjust **accumulate** and **batch_size**.

.. code-block:: console

    $ python finetune_squad.py --optimizer bertadam --accumulate 2 --batch_size 6 --gpu


Should produce an output like this. Explain that the F1 score on the dev dataset is 88.45%

.. code-block:: console

    {'exact_match': 81.21097445600756, 'f1': 88.4551346176558}
