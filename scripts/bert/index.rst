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

GluonNLP provides the following example script to fine-tune SQuAD with pre-trained BERT model.

The throughputs of training and inference are based on fixed sequence length=384 and input embedding size=768, which are 1.65 samples/s and 3.97 samples/s respectively.

In total, one training epoch costs 4466.87s and inference costs 113.99s on SQuAD v1.1.

The evaluation result of the model after one training epoch is 'Exact Match': 78.78, 'F1': 86.99.

To reproduce the above result, simply run the following command with MXNet==1.5.0b20190116.
 
.. code-block:: console

    $ python finetune_squad.py --optimizer adam --gpu
