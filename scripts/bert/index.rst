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

BERT for Sentence Classification on GLUE tasks and XNLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example script to fine-tune sentence classification with pre-trained
BERT model.

+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|                     |                                                  MRPC                                                  |                                                  RTE                                                  |                                                 SST-2                                                 |                                                MNLI-m/mm                                               |                                             XNLI (chinese)                                             |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|        model        |                                             bert_12_768_12                                             |                                             bert_12_768_12                                            |                                             bert_12_768_12                                            |                                             bert_12_768_12                                             |                                             bert_12_768_12                                             |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| validation accuracy |                                                  88.7%                                                 |                                                 70.8%                                                 |                                                  93%                                                  |                                             84.55%, 84.66%                                             |                                                 78.27%                                                 |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|      batch_size     |                                                   32                                                   |                                                   32                                                  |                                                   16                                                  |                                                   32                                                   |                                                   32                                                   |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|        epochs       |                                                    3                                                   |                                                   3                                                   |                                                   4                                                   |                                                    3                                                   |                                                    4                                                   |
|                     |                                                                                                        |                                                                                                       |                                                                                                       |                                                                                                        |                                                                                                        |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|       epsilon       |                                                  1e-6                                                  |                                                  1e-6                                                 |                                                  1e-6                                                 |                                                  1e-8                                                  |                                                  1e-6                                                  |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|     training log    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mrpc.log>`__ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_rte.log>`__ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_sst.log>`__ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mnli.log>`__ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_xnli.log>`__ |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+
|       command       |                                                   [1]                                                  |                                                  [2]                                                  |                                                  [3]                                                  |                                                   [4]                                                  |                                                   [5]                                                  |
+---------------------+--------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------+

For all model settings above, we set learing rate = 2e-5 and optimizer = bertadam.

[1] MRPC

.. code-block:: console

    $ # download the dataset from https://www.microsoft.com/en-us/download/details.aspx?id=52398 and unzip it to ./MRPC
    $ python finetune_classifier.py --task_name MRPC --batch_size 32 --epochs 3 --gpu 0 --lr 2e-5

[2] SST-2

.. code-block:: console

    $ python finetune_classifier.py --task_name SST --epochs 4 --batch_size 16 --gpu 0 --lr 2e-5 --log_interval 500

[3] RTE

.. code-block:: console

    $ python finetune_classifier.py --task_name RTE --batch_size 32 --epochs 3 --gpu 0 --lr 2e-5

[4] MNLI

.. code-block:: console

    $ python finetune_classifier.py --task_name MNLI --max_len 80 --log_interval 100 --epsilon 1e-8 --gpu 0

[5] XNLI (chinese)

.. code-block:: console

    $ BAIDU_ERNIE_DATA_DIR=baidu_ernie_data python finetune_classifier.py --seed 6 --task_name XNLI --batch_size 32 --optimizer bertadam --epochs 4 --lr 2e-5 --bert_dataset wiki_cn_cased --gpu 0

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

Training Sample Generation
++++++++++++++++++++++++++

Data generation for pre-training on sample texts:

.. code-block:: console

    $ python create_pretraining_data.py --input_file sample_text.txt --output_dir out --vocab book_corpus_wiki_en_uncased --max_seq_length 128 --max_predictions_per_seq 20 --dupe_factor 5 --masked_lm_prob 0.15 --short_seq_prob 0.1 --verbose

The data generation script takes a file path as the input (could be one or more files by wildcard). Each file contains one or more documents separated by empty lines, and each document contains one line per sentence. You can perform sentence segmentation with an off-the-shelf NLP toolkit such as NLTK.

Run Pre-training
++++++++++++++++

Run pre-training with generated data:

.. code-block:: console

    $ python run_pretraining.py --gpus 0 --batch_size 32 --lr 2e-5 --data 'out/*.npz' --warmup_ratio 0.5 --num_steps 20 --pretrained --log_interval=2 --data_eval 'out/*.npz' --batch_size_eval 8 --ckpt_dir ckpt --verbose

With 20 steps of pre-training it easily reaches above 90% masked language model accuracy and 98% next sentence prediction accuracy on the training data.

To reproduce BERT pre-training with books corpus and English wikipedia datasets from scratch, we recommend using float16 for pre-training with gradient accumulation.

.. code-block:: console

    $ python run_pretraining.py --gpus 0,1,2,3,4,5,6,7 --batch_size 8 --accumulate 4 --lr 1e-4 --data '/path/to/generated/samples/train/*.npz' --warmup_ratio 0.01 --num_steps 1000000 --log_interval=250 --data_eval '/path/to/generated/samples/dev/*.npz' --batch_size_eval 8 --ckpt_dir ckpt --ckpt_interval 25000 --num_buckets 10 --dtype float16

The BERT base model produced by gluonnlp pre-training script (`log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/bert_base_pretrain.log>`__) achieves 83.6% on MNLI-mm, 93% on SST-2, 87.99% on MRPC and 80.99/88.60 on SQuAD 1.1 validation set.

Run Pre-training with Horovod
+++++++++++++++++++++++++++++

Alternatively, you can install horovod for scalable multi-gpu multi-machine training. Our script assumes the master version of Horovod (i.e. horovod > v0.16.1).

To install horovod, you need:

- `NCCL <https://developer.nvidia.com/nccl>`__, and
- `OpenMPI <https://www.open-mpi.org/software/ompi/v4.0/>`__

Then you can install the master version of horovod:

.. code-block:: console

    $ git clone --recursive https://github.com/uber/horovod horovod;
    $ cd horovd;
    $ HOROVOD_GPU_ALLREDUCE=NCCL pip install . --user --no-cache-dir

Verify Horovod installation:

.. code-block:: console

    $ horovodrun -np 1 -H localhost:1 python run_pretraining_hvd.py --batch_size 32 --lr 2e-5 --data 'out/*.npz' --warmup_ratio 0.5 --num_steps 20 --pretrained --log_interval=2 --data_eval 'out/*.npz' --batch_size_eval 8 --ckpt_dir ckpt --verbose

Run pre-training with horovod:

.. code-block:: console

    $ horovodrun -np 8 -H localhost:8 python run_pretraining_hvd.py --data='/path/to/generated/samples/train/*.npz' --num_steps 1000000 --log_interval 250 --lr 1e-4 --batch_size 4096 --accumulate 4 --warmup_ratio 0.01 --ckpt_dir ./ckpt --ckpt_interval 25000 --num_buckets 10 --dtype float16 --use_avg_len --verbose
    $ mpirun -np 16 -H node0:8,node1:8 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo --map-by ppr:4:socket -x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO python run_pretraining_hvd.py --batch_size 8192 --accumulate 1 --lr 1e-4 --data "/path/to/generated/samples/train/*.npz" --warmup_ratio 0.01 --num_steps 1000000 --log_interval=250 --ckpt_dir './ckpt' --ckpt_interval 25000 --num_buckets 10 --dtype float16 --use_avg_len --verbose

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


Export BERT for Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Current export/export.py support exporting BERT models. Supported values for --task argument include classification, regression and question_answering.

.. code-block:: console

    $ python export/export.py --task classification --model_parameters /path/to/saved/ckpt.params --output_dir /path/to/output/dir/ --seq_length 128

This will export the BERT model for classification to a symbol.json file, saved to the directory specified by --output_dir.
The --model_parameters argument is optional. If not set, the .params file saved in the output directory will be randomly intialized parameters.
