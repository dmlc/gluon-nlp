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

BERT for Sentence or Tokens Embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goal of this BERT Embedding is to obtain the sentence and token embedding from BERT's pre-trained model. In this way, instead of building and do fine-tuning for an end-to-end NLP model, you can build your model by just utilizing the sentence or token embeddings.

Usage
"""""

.. code-block:: python
    from bert.bert_embedding import BertEmbedding

    bert_abstract = """We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.
     Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
     As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
    BERT is conceptually simple and empirically powerful.
    It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7 (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5% absolute improvement), outperforming human performance by 2.0%."""
    sentences = bert_abstract.split('\n')
    bert = BertEmbedding()
    result = bert.embedding(sentences)

If you want to use GPU, please import mxnet and set context

.. code-block:: python
    import mxnet as mx
    from bert.bert_embedding import BertEmbedding

    ...

    ctx = mx.gpu(0)
    bert = BertEmbedding(ctx=ctx)

Example of using the large pre-trained BERT model from Google

.. code-block:: python
    from bert.bert_embedding import BertEmbedding

    bert = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')

Example outputs:

.. code-block:: python
    first_sentence = result[0]

    first_sentence[0]
    # array([-0.835946  , -0.4605566 , -0.95620036, ..., -0.95608854,
    #       -0.6258104 ,  0.7697007 ], dtype=float32)
    first_sentence[0].shape
    # (768,)

    first_sentence[1]
    # ['we', 'introduce', 'a', 'new', 'language', 'representation', 'model', 'called', 'bert', ',', 'which', 'stands', 'for', 'bidirectional', 'encoder', 'representations', 'from', 'transformers']
    len(first_sentence[1])
    # 18


    len(first_sentence[2])
    # 18
    first_token_in_first_sentence = first_sentence[2]
    first_token_in_first_sentence[0]
    # array([ 0.4805648 ,  0.18369392, -0.28554988, ..., -0.01961522,
    #        1.0207764 , -0.67167974], dtype=float32)
    first_token_in_first_sentence[0].shape
    # (768,)
