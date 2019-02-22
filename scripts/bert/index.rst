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

Some other tasks can be modeled with `--task_name` parameter.

.. code-block:: console

   $ MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=glue_data python3 finetune_classifier.py --task_name MNLI --max_len 80 --log_interval 100 --epsilon 1e-8 --gpu

It gets MNLI validation accuracy ,On dev_matched.tsv: 84.6%
On dev_mismatched.tsv: 84.7%. `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetuned_mnli.log>`_


Some other tasks can be modeled with `--task_name` parameter.


BERT for SQuAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example script to fine-tune SQuAD with pre-trained
BERT model.

SQuAD 1.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the following command to fine-tune the BERT base model for SQuAD1.1 dataset.

Note that this will require more than 12G of GPU memory.
 
.. code-block:: console

    $ python finetune_squad.py --optimizer adam --batch_size 12 --lr 3e-5 --epochs 2 --gpu

python finetune_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 6 --batch_size 4 --lr 3e-5 --epochs 2 --gpu

If you are using less than 12G of GPU memory, you can use the following command to achieve a similar effect.

Note that this will require approximately no more than 8G of GPU memory. If your GPU memory is too small, you can adjust **accumulate** and **batch_size**.

.. code-block:: console

    $ python finetune_squad.py --optimizer adam --accumulate 2 --batch_size 6 --lr 3e-5 --epochs 2 --gpu

The F1 score on the dev dataset is `88.45% <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_base_mx1.5.0b20190216.log>`_ (Based on mxnet-cu90-1.5.0b20190216)

Use the following command to fine-tune the BERT large model for SQuAD1.1 dataset.

Note that this will require more than 14G of GPU memory.

.. code-block:: console

    $ python finetune_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 6 --batch_size 4 --lr 3e-5 --epochs 2 --gpu

The F1 score on the dev dataset is `90.97% <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_large_mx1.5.0b20190216.log>`_ (Based on mxnet-cu90-1.5.0b20190216)


SQuAD 2.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are pre-training on the SQuAD2.0 dataset, you need to specify the parameter **version_2** and specify the parameter **null_score_diff_threshold** (Typical values are between -1.0 and -5.0).

Use the following command to fine tune the BERT large model of the SQuAD2.0 dataset and generate `predictions.json, nbest_predictions.json, and null_odds.json. <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad2.0_large_mx1.5.0b20160216.log>`_ (Based on mxnet-cu90-1.5.0b20190216)

.. code-block:: console

    $ python finetune_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 8 --batch_size 4 --lr 3e-5 --epochs 2 --gpu --null_score_diff_threshold -2.0 --version_2

If you want to get the score of the dev data, you need to download the dev dataset and the evaluate script.

`dev-v2.0.json <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json>`_ 

`evaluate-2.0.py <https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/>`_

Use the following command to get the score of the dev dataset

.. code-block:: console

    $ python evaluate-v2.0.py dev-v2.0.json predictions.json

Using the predictions.json file generated above, the result should look like this:

.. code-block:: json
    
    {
        "exact": 77.958392992504,
        "f1": 81.02012658815627,
        "total": 11873,
        "HasAns_exact": 73.3974358974359,
        "HasAns_f1": 79.52968336389662,
        "HasAns_total": 5928,
        "NoAns_exact": 82.50630782169891,
        "NoAns_f1": 82.50630782169891,
        "NoAns_total": 5945
    }

