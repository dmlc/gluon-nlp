Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

:download:`[Download] </model_zoo/bert.zip>`

Reference: Devlin, Jacob, et al. "`Bert: Pre-training of deep bidirectional transformers for language understanding. <https://arxiv.org/abs/1810.04805>`_" arXiv preprint arXiv:1810.04805 (2018).

Note: BERT model requires `nightly version of MXNet <https://mxnet.incubator.apache.org/versions/master/install/index.html?version=master&platform=Linux&language=Python&processor=CPU>`__. 

The following pre-trained BERT models are available from the **gluonnlp.model.get_model** API:

+-----------------------------+----------------+-----------------+
|                             | bert_12_768_12 | bert_24_1024_16 |
+=============================+================+=================+
| book_corpus_wiki_en_uncased | ✓              | ✓               |
+-----------------------------+----------------+-----------------+
| book_corpus_wiki_en_cased   | ✓              | ✓               |
+-----------------------------+----------------+-----------------+
| wiki_multilingual_uncased   | ✓              | x               |
+-----------------------------+----------------+-----------------+
| wiki_multilingual_cased     | ✓              | x               |
+-----------------------------+----------------+-----------------+
| wiki_cn_cased               | ✓              | x               |
+-----------------------------+----------------+-----------------+

where **bert_12_768_12** refers to the BERT BASE model, and **bert_24_1024_16** refers to the BERT LARGE model.

BERT for Sentence Classification on GLUE tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example script to fine-tune sentence classification with pre-trained
BERT model.

+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|                     |                                                  MRPC                                                  |                                                  RTE                                                  |                                                 SST-2                                                 |                                                MNLI-m/mm                                               |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|        model        |                                             bert_12_768_12                                             |                                             bert_12_768_12                                            |                                             bert_12_768_12                                            |                                             bert_12_768_12                                             |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| validation accuracy |                                                  88.7%                                                 |                                                 70.8%                                                 |                                                  93%                                                  |                                             84.55%, 84.66%                                             |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|      batch_size     |                                                   32                                                   |                                                   32                                                  |                                                   16                                                  |                                                   32                                                   |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|        epochs       |                                                    3                                                   |                                                   3                                                   |                                                   4                                                   |                                                    3                                                   |
|                     |                                                                                                        |                                                                                                       |                                                                                                       |                                                                                                        |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|       epsilon       |                                                  1e-6                                                  |                                                  1e-6                                                 |                                                  1e-6                                                 |                                                  1e-8                                                  |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|     training log    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mrpc.log>`__ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_rte.log>`__ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_sst.log>`__ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mnli.log>`__ |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|       command       |                                                   [1]                                                  |                                                  [2]                                                  |                                                  [3]                                                  |                                                   [4]                                                  |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+

For all model settings above, we set learing rate = 2e-5 and optimizer = bertadam.

[1] MRPC

.. code-block:: console

    $ curl -L https://tinyurl.com/yaznh3os -o download_glue_data.py
    $ python3 download_glue_data.py --data_dir glue_data --tasks MRPC
    $ GLUE_DIR=glue_data python finetune_classifier.py --task_name MRPC --batch_size 32 --optimizer bertadam --epochs 3 --gpu --lr 2e-5

[2] SST-2

.. code-block:: console

    $ curl -L https://tinyurl.com/yaznh3os -o download_glue_data.py
    $ python3 download_glue_data.py --data_dir glue_data --tasks SST
    $ GLUE_DIR=glue_data python finetune_classifier.py --task_name SST --epochs 4 --batch_size 16 --optimizer bertadam --gpu --lr 2e-5 --log_interval 500

[3] RTE

.. code-block:: console

    $ curl -L https://tinyurl.com/yaznh3os -o download_glue_data.py
    $ python3 download_glue_data.py --data_dir glue_data --tasks RTE
    $ GLUE_DIR=glue_data python finetune_classifier.py --task_name RTE --batch_size 32 --optimizer bertadam --epochs 3 --gpu  --lr 2e-5

[4] MNLI

.. code-block:: console

    $ curl -L https://tinyurl.com/yaznh3os -o download_glue_data.py
    $ python3 download_glue_data.py --data_dir glue_data --tasks MNLI
    $ GLUE_DIR=glue_data python finetune_classifier.py --task_name MNLI --max_len 80 --log_interval 100 --epsilon 1e-8 --gpu

Some other tasks can be modeled with `--task_name` parameter.

BERT for Question Answering on SQuAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
|                       |                                                            SQuAD 1.1                                                            |                                                             SQuAD 1.1                                                            |                                                             SQuAD 2.0                                                            |
+-----------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
|         model         |                                                          bert_12_768_12                                                         |                                                          bert_24_1024_16                                                         |                                                          bert_24_1024_16                                                         |
+-----------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
|           F1          |                                                              88.53                                                              |                                                               90.97                                                              |                                                               77.96                                                              |
+-----------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
|           EM          |                                                              80.98                                                              |                                                               84.05                                                              | 81.02                                                                                                                            |
+-----------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
|       batch_size      |                                                                12                                                               |                                                                 4                                                                |                                                                 4                                                                |
+-----------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| gradient accumulation |                                                               None                                                              |                                                                 6                                                                |                                                                 8                                                                |
+-----------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
|         epochs        |                                                                2                                                                |                                                                 2                                                                |                                                                 2                                                                |
+-----------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
|      training log     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_base_mx1.5.0b20190216.log>`__ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_large_mx1.5.0b20190216.log>`__ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad2.0_large_mx1.5.0b20160216.log>`__ |
+-----------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
|        command        |                                                               [5]                                                               |                                                                [6]                                                               |                                                                [7]                                                               |
+-----------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+

For all model settings above, we set learing rate = 3e-5 and optimizer = adam.

BERT BASE on SQuAD 1.1
++++++++++++++++++++++

[5] bert_12_768_12

.. code-block:: console

    $ python finetune_squad.py --optimizer adam --batch_size 12 --lr 3e-5 --epochs 2 --gpu
 
Note that this requires about 12G of GPU memory. If your GPU memory is less than 12G, you can use the following command to achieve a similar effect. This will require approximately no more than 8G of GPU memory. If your GPU memory is too small, please adjust *accumulate* and *batch_size* arguments accordingly.

.. code-block:: console

    $ python finetune_squad.py --optimizer adam --accumulate 2 --batch_size 6 --lr 3e-5 --epochs 2 --gpu


BERT LARGE on SQuAD 1.1
+++++++++++++++++++++++

[6] bert_24_1024_16

.. code-block:: console

    $ python finetune_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 6 --batch_size 4 --lr 3e-5 --epochs 2 --gpu
    
Note that this requires about 14G of GPU memory.


BERT LARGE on SQuAD 2.0
+++++++++++++++++++++++

For SQuAD 2.0, you need to specify the parameter *version_2* and specify the parameter *null_score_diff_threshold*. Typical values are between -1.0 and -5.0. Use the following command to fine-tune the BERT large model on SQuAD 2.0 and generate predictions.json, nbest_predictions.json, and null_odds.json.

[7] bert_24_1024_16

.. code-block:: console

    $ python finetune_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 8 --batch_size 4 --lr 3e-5 --epochs 2 --gpu --null_score_diff_threshold -2.0 --version_2

To get the score of the dev data, you need to download the dev dataset (`dev-v2.0.json <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json>`_) and the evaluate script (`evaluate-2.0.py <https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/>`_). Then use the following command to get the score of the dev dataset.

.. code-block:: console

    $ python evaluate-v2.0.py dev-v2.0.json predictions.json

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

BERT Pre-training
~~~~~~~~~~~~~~~~~

The scripts for masked language modeling and and next sentence prediction are also provided.

Data generation for pre-training on sample texts:

.. code-block:: console

    $ python create_pretraining_data.py --input_file sample_text.txt --output_dir out --vocab book_corpus_wiki_en_uncased --max_seq_length 128 --max_predictions_per_seq 20 --dupe_factor 5 --masked_lm_prob 0.15 --short_seq_prob 0.1 --verbose

The data generation script takes a file path as the input (could be one or more files by wildcard). Each file contains one or more documents separated by empty lines, and each document contains one line per sentence. You can perform sentence segmentation with an off-the-shelf NLP toolkit such as NLTK.

Run pre-training with generated data:

.. code-block:: console

    $ python run_pretraining.py --gpus 0 --batch_size 32 --lr 2e-5 --data 'out/*.npz' --warmup_ratio 0.5 --num_steps 20 --pretrained --log_interval=2 --data_eval 'out/*.npz' --batch_size_eval 8 --ckpt_dir ckpt

With 20 steps of pre-training it easily reaches above 90% masked language model accuracy and 98% next sentence prediction accuracy on the training data.

To reproduce BERT pre-training with books corpus and English wikipedia datasets from scratch, we recommend using float16 for pre-training with gradient accumulation.

.. code-block:: console

    $ python run_pretraining.py --gpus 0,1,2,3,4,5,6,7 --batch_size 8 --lr 1e-4 --data '/path/to/generated/samples/train/*.npz' --warmup_ratio 0.01 --num_steps 1000000 --log_interval=250 --data_eval '/path/to/generated/samples/dev/*.npz' --batch_size_eval 8 --ckpt_dir ckpt --ckpt_interval 25000 --accumulate 4 --num_buckets 10 --dtype float16

The BERT base model produced by gluonnlp pre-training script (`log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/bert_base_pretrain.log>`__) achieves 83.6% on MNLI-mm, 93% on SST-2, 87.99% on MRPC and 80.99/88.60 on SQuAD 1.1 validation set.

BERT for Sentence or Tokens Embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goal of this BERT Embedding is to obtain the token embedding from BERT's pre-trained model. In this way, instead of building and do fine-tuning for an end-to-end NLP model, you can build your model by just utilizing the token embeddings.

Usage
+++++

.. code-block:: python

    from bert.embedding import BertEmbedding

    bert_abstract = """We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.
     Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
     As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
    BERT is conceptually simple and empirically powerful.
    It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7 (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5% absolute improvement), outperforming human performance by 2.0%."""
    sentences = bert_abstract.split('\n')
    bert_embedding = BertEmbedding()
    result = bert_embedding(sentences)

If you want to use GPU, please import mxnet and set context

.. code-block:: python

    import mxnet as mx
    from bert.embedding import BertEmbedding

    ctx = mx.gpu(0)
    bert_embedding = BertEmbedding(ctx=ctx)

Example of using the large pre-trained BERT model from Google

.. code-block:: python

    from bert.embedding import BertEmbedding

    bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')

Example outputs:

.. code-block:: python

    first_sentence = result[0]

    first_sentence[0]
    # ['we', 'introduce', 'a', 'new', 'language', 'representation', 'model', 'called', 'bert', ',', 'which', 'stands', 'for', 'bidirectional', 'encoder', 'representations', 'from', 'transformers']
    len(first_sentence[0])
    # 18


    len(first_sentence[1])
    # 18
    first_token_in_first_sentence = first_sentence[1]
    first_token_in_first_sentence[0]
    # array([ 0.4805648 ,  0.18369392, -0.28554988, ..., -0.01961522,
    #        1.0207764 , -0.67167974], dtype=float32)
    first_token_in_first_sentence[0].shape
    # (768,)

Command line interface
++++++++++++++++++++++

.. code-block:: shell

    python bert_embedding/bert.py --sentences "GluonNLP is a toolkit that enables easy text preprocessing, datasets loading and neural models building to help you speed up your Natural Language Processing (NLP) research."
    Text: GluonNLP is a toolkit that enables easy text preprocessing, datasets loading and neural models building to help you speed up your Natural Language Processing (NLP) research.
    Tokens embedding: [array([-0.11881411, -0.59530115,  0.627092  , ...,  0.00648153,
       -0.03886228,  0.03406909], dtype=float32), array([-0.7995638 , -0.6540758 , -0.00521846, ..., -0.42272145,
       -0.5787281 ,  0.7021201 ], dtype=float32), array([-0.7406778 , -0.80276626,  0.3931962 , ..., -0.49068323,
       -0.58128357,  0.6811132 ], dtype=float32), array([-0.43287313, -1.0018158 ,  0.79617643, ..., -0.26877284,
       -0.621779  , -0.2731115 ], dtype=float32), array([-0.8515188 , -0.74098676,  0.4427735 , ..., -0.41267148,
       -0.64225197,  0.3949393 ], dtype=float32), array([-0.86652845, -0.27746758,  0.8806506 , ..., -0.87452525,
       -0.9551989 , -0.0786318 ], dtype=float32), array([-1.0987284 , -0.36603633,  0.2826037 , ..., -0.33794224,
       -0.55210876, -0.09221527], dtype=float32), array([-0.3483025 ,  0.401534  ,  0.9361341 , ..., -0.29747447,
       -0.49559578, -0.08878893], dtype=float32), array([-0.65626   , -0.14857645,  0.29733548, ..., -0.15890433,
       -0.45487815, -0.28494897], dtype=float32), array([-0.1983894 ,  0.67196256,  0.7867421 , ..., -0.7990434 ,
        0.05860569, -0.26884627], dtype=float32), array([-0.3775159 , -0.00590206,  0.5240432 , ..., -0.26754653,
       -0.37806216,  0.23336883], dtype=float32), array([ 0.1876977 ,  0.30165672,  0.47167772, ..., -0.43823618,
       -0.42823148, -0.48873612], dtype=float32), array([-0.6576557 , -0.09822252,  0.1121515 , ..., -0.21743725,
       -0.1820574 , -0.16115054], dtype=float32)]


Example Usage of Exporting Hybridizable BERT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The BERTModel class is a subclass of Block, rather than HybridBlock.
To support exporting BERT model to json format for deployment, we introduce the StaticBERT class.
Specifically, by exporting hybridizable BERT, we mean the BERT with fixed input embedding size and sequence length can be exported through
a static shape based implementation of hybridblock based BERT. By using this, we can export a block based BERT model.

Please follow the steps below for exporting the model.


Step 1: create a hybridizable task guided model using BERT:

.. code-block:: python

    class StaticBertForQA(HybridBlock)

An example can be found in 'staticbert/static_bert_for_qa_model.py'.


Step 2: hybridize the model in the script:

.. code-block:: python

    net = StaticBertForQA(bert=bert)
    net.hybridize(static_alloc=True, static_shape=True)

An example can be found in 'staticbert/static_export_squad.py'.


Step 3: export trained model:

.. code-block:: python

    net.export(os.path.join(args.output_dir, 'static_net'), epoch=args.epochs)

To export the model, in 'staticbert/static_export_squad.py', set export=True.

To run the example, if you would like to export the Block parameters
and test the HybridBlock on your datasets with the specified input size and sequence length,

.. code-block:: console

    $ cd staticbert
    $ python static_export_squad.py --model_parameters output_dir/net.params --export --evaluate --seq_length 384 --input_size 768 --gpu 0

This will load the the StaticBERTQA HybridBlock with parameter (requirement: output_dir/net.params should exist)
trained by a normal BERTQA Block, and export the HybridBlock to json format.

Besides, Where seq_length stands for the sequence length of the input, input_size represents the embedding size of the input.


Example Usage of Finetuning Hybridizable BERT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example mainly introduces the steps needed to use the hybridizable BERT models to finetune on a specific NLP task.
We use SQuAD dataset for Question Answering as an example.


Step 1-3 are the same as in previous section 'Example Usage of Exporting Hybridizable BERT',
where an example of Step 1 can be found in 'staticbert/static_bert_for_qa_model.py',
an example of Step 2-3 can be found in 'staticbert/static_finetune_squad.py'.
To export the model, in 'staticbert/static_finetune_squad.py', set export=True.


For all model settings above, we set learning rate = 3e-5 and optimizer = adam.
Besides, seq_length stands for the sequence length of the input, input_size represents the embedding size of the input.
The options can be specified in the following command lines.


+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
|                       | SQuAD 1.1                                                                                                                  | SQuAD 1.1                                                                                                                   | SQuAD 2.0                                                                                                                   |
+=======================+============================================================================================================================+=============================================================================================================================+=============================================================================================================================+
| model                 | bert_12_768_12                                                                                                             | bert_24_1024_16                                                                                                             | bert_24_1024_16                                                                                                             |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| F1                    | 88.54                                                                                                                      | 90.84                                                                                                                       | 81.46                                                                                                                       |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| EM                    | 81.10                                                                                                                      | 84.03                                                                                                                       | 78.49                                                                                                                       |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| batch_size            | 12                                                                                                                         | 4                                                                                                                           | 4                                                                                                                           |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| gradient accumulation | None                                                                                                                       | 6                                                                                                                           | 8                                                                                                                           |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| epochs                | 2                                                                                                                          | 2                                                                                                                           | 2                                                                                                                           |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| training log          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/static_finetune_squad1.1_base.log>`__      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/static_finetune_squad1.1_large.log>`__      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/static_finetune_squad2.0_large.log>`__      |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| command               | [8]                                                                                                                        | [9]                                                                                                                         | [10]                                                                                                                        |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+

BERT BASE on SQuAD 1.1
++++++++++++++++++++++

[8] bert_12_768_12

.. code-block:: console

    $ cd staticbert
    $ python static_finetune_squad.py --optimizer adam --batch_size 12 --lr 3e-5 --epochs 2 --gpu 0 --export


BERT LARGE on SQuAD 1.1
+++++++++++++++++++++++

[9] bert_24_1024_16

.. code-block:: console

    $ cd staticbert
    $ python static_finetune_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 6 --batch_size 4 --lr 3e-5 --epochs 2 --gpu 0 --export


BERT LARGE on SQuAD 2.0
+++++++++++++++++++++++

[10] bert_24_1024_16

.. code-block:: console

    $ cd staticbert
    $ python static_finetune_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 8 --batch_size 4 --lr 3e-5 --epochs 2 --gpu 0 --null_score_diff_threshold -2.0 --version_2 --export

To get the score of the dev data, you need to download the dev dataset (`dev-v2.0.json <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json>`_) and the evaluate script (`evaluate-2.0.py <https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/>`_). Then use the following command to get the score of the dev dataset.

.. code-block:: console

    $ cd staticbert
    $ python evaluate-v2.0.py dev-v2.0.json predictions.json

.. code-block:: json

    {
        "exact": 78.49743114629833,
        "f1": 81.46366127573552,
        "total": 11873,
        "HasAns_exact": 73.38056680161944,
        "HasAns_f1": 79.32153345593925,
        "HasAns_total": 5928,
        "NoAns_exact": 83.59966358284272,
        "NoAns_f1": 83.59966358284272,
        "NoAns_total": 5945
    }
