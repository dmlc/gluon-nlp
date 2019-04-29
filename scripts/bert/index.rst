Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

:download:`[Download] </model_zoo/bert.zip>`

Reference: Devlin, Jacob, et al. "`Bert: Pre-training of deep bidirectional transformers for language understanding. <https://arxiv.org/abs/1810.04805>`_" arXiv preprint arXiv:1810.04805 (2018).

The following pre-trained BERT models are available from the **gluonnlp.model.get_model** API:

+--------------------+---------------------------------+-------------------------------+----------------------------+-------------------------+---------------+
|                    | book_corpus_wiki_en_uncased     | book_corpus_wiki_en_cased     | wiki_multilingual_uncased  | wiki_multilingual_cased | wiki_cn_cased |
+====================+=================================+===============================+============================+=========================+===============+
| bert_12_768_12     | ✓                               | ✓                             | ✓                          | ✓                       | ✓             |
+--------------------+---------------------------------+-------------------------------+----------------------------+-------------------------+---------------+
| bert_24_1024_16    | ✓                               | ✓                             | x                          | x                       | x             |
+--------------------+---------------------------------+-------------------------------+----------------------------+-------------------------+---------------+

where **bert_12_768_12** refers to the BERT BASE model, and **bert_24_1024_16** refers to the BERT LARGE model.

BERT for Sentence Classification on GLUE tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example script to fine-tune sentence classification with pre-trained
BERT model.

Download the GLUE dataset:

 .. code-block:: console

    $ curl -L https://tinyurl.com/yaznh3os -o download_glue_data.py
    $ python3 download_glue_data.py --data_dir glue_data --tasks all

Use the following command to fine-tune the BERT model for classification on the GLUE(MRPC, RTE, QQP, QNLI, STS-B, CoLA, MNLI, WNLI, SST) dataset.

.. code-block:: console

   $ MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=glue_data finetune_classifier.py --task_name MRPC --batch_size 32 --optimizer bertadam --epochs 3 --gpu --lr 2e-5

It gets validation accuracy of `88.7% <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mrpc.log>`_, whereas the the original Tensorflow implementation give evaluation results between 84% and 88%.

.. code-block:: console

   $ MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=glue_data python3 finetune_classifier.py --task_name RTE --batch_size 32 --optimizer bertadam --epochs 3 --gpu  --lr 2e-5

It gets RTE validation accuracy of `70.8% <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_rte.log>`_
, whereas the the original Tensorflow implementation give evaluation results 66.4%.

.. code-block:: console

   $ MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=glue_data python3 finetune_classifier.py --task_name MNLI --max_len 80 --log_interval 100 --epsilon 1e-8 --gpu

It gets MNLI validation accuracy ,On dev_matched.tsv: 84.6%
On dev_mismatched.tsv: 84.7%. `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetuned_mnli.log>`_


Some other tasks can be modeled with `--task_name` parameter.

BERT for Named Entity Recognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides training and prediction script for named entity recognition models.
Dataset should be formatted in `CoNLL-2003 shared task format <https://www.clips.uantwerpen.be/conll2003/ner/>`_.
Assuming data files are located in `${DATA_DIR}`, below command trains BERT model for
named entity recognition, and saves model artifacts to `${MODEL_DIR}` with `large_bert`
prefix in file names:

 .. code-block:: console

    $ python3 train_ner.py \
        --train-path ${DATA_DIR}/train.txt \
        --dev-path ${DATA_DIR}/dev.txt \
        --test-path ${DATA_DIR}/test.txt
        --gpu 0 --learning-rate 1e-5 --dropout-prob 0.1 --num-epochs 100 --batch-size 8 \
        --optimizer bertadam --bert-model bert_24_1024_16 \
        --save-checkpoint-prefix ${MODEL_DIR}/large_bert --seed 13531

This achieves Test F1 from `91.5` to `92.2`.
