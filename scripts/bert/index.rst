BERT
----

:download:`Download scripts </model_zoo/bert.zip>`


Reference: Devlin, Jacob, et al. "`Bert: Pre-training of deep bidirectional transformers for language understanding. <https://arxiv.org/abs/1810.04805>`_" arXiv preprint arXiv:1810.04805 (2018).

BERT Model Zoo
~~~~~~~~~~~~~~

The following pre-trained BERT models are available from the **gluonnlp.model.get_model** API:

+-----------------------------------------+----------------+-----------------+
|                                         | bert_12_768_12 | bert_24_1024_16 |
+=========================================+================+=================+
| book_corpus_wiki_en_uncased             | ✓              | ✓               |
+-----------------------------------------+----------------+-----------------+
| book_corpus_wiki_en_cased               | ✓              | ✓               |
+-----------------------------------------+----------------+-----------------+
| openwebtext_book_corpus_wiki_en_uncased | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| wiki_multilingual_uncased               | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| wiki_multilingual_cased                 | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| wiki_cn_cased                           | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| scibert_scivocab_uncased                | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| scibert_scivocab_cased                  | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| scibert_basevocab_uncased               | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| scibert_basevocab_cased                 | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| biobert_v1.0_pmc_cased                  | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| biobert_v1.0_pubmed_cased               | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| biobert_v1.0_pubmed_pmc_cased           | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| biobert_v1.1_pubmed_cased               | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| clinicalbert_uncased                    | ✓              | x               |
+-----------------------------------------+----------------+-----------------+
| kobert_news_wiki_ko_cased               | ✓              | x               |
+-----------------------------------------+----------------+-----------------+

where **bert_12_768_12** refers to the BERT BASE model, and **bert_24_1024_16** refers to the BERT LARGE model.

.. code-block:: python

    import gluonnlp as nlp; import mxnet as mx;
    model, vocab = nlp.model.get_model('bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased', use_classifier=False, use_decoder=False);
    tokenizer = nlp.data.BERTTokenizer(vocab, lower=True);
    transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=512, pair=False, pad=False);
    sample = transform(['Hello world!']);
    words, valid_len, segments = mx.nd.array([sample[0]]), mx.nd.array([sample[1]]), mx.nd.array([sample[2]]);
    seq_encoding, cls_encoding = model(words, segments, valid_len);


The pretrained parameters for dataset_name
'openwebtext_book_corpus_wiki_en_uncased' were obtained by running the GluonNLP
BERT pre-training script on OpenWebText.

The pretrained parameters for dataset_name 'scibert_scivocab_uncased',
'scibert_scivocab_cased', 'scibert_basevocab_uncased', 'scibert_basevocab_cased'
were obtained by converting the parameters published by "Beltagy, I., Cohan, A.,
& Lo, K. (2019). Scibert: Pretrained contextualized embeddings for scientific
text. arXiv preprint `arXiv:1903.10676 <https://arxiv.org/abs/1903.10676>`_."

The pretrained parameters for dataset_name 'biobert_v1.0_pmc',
'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc', 'biobert_v1.1_pubmed' were
obtained by converting the parameters published by "Lee, J., Yoon, W., Kim, S.,
Kim, D., Kim, S., So, C. H., & Kang, J. (2019). Biobert: pre-trained biomedical
language representation model for biomedical text mining. arXiv preprint
`arXiv:1901.08746 <https://arxiv.org/abs/1901.08746>`_."

The pretrained parameters for dataset_name 'clinicalbert' were obtained by
converting the parameters published by "Huang, K., Altosaar, J., & Ranganath, R.
(2019). ClinicalBERT: Modeling Clinical Notes and Predicting Hospital
Readmission. arXiv preprint `arXiv:1904.05342
<https://arxiv.org/abs/1904.05342>`_."

Additionally, GluonNLP supports the "`RoBERTa <https://arxiv.org/abs/1907.11692>`_" model:

+-----------------------------------------+-------------------+--------------------+
|                                         | roberta_12_768_12 | roberta_24_1024_16 |
+=========================================+===================+====================+
| openwebtext_ccnews_stories_books_cased  | ✓                 | ✓                  |
+-----------------------------------------+-------------------+--------------------+

.. code-block:: python

    import gluonnlp as nlp; import mxnet as mx;
    model, vocab = nlp.model.get_model('roberta_12_768_12', dataset_name='openwebtext_ccnews_stories_books_cased', use_decoder=False);
    tokenizer = nlp.data.GPT2BPETokenizer();
    text = [vocab.bos_token] + tokenizer('Hello world!') + [vocab.eos_token];
    seq_encoding = model(mx.nd.array([vocab[text]]))

GluonNLP also supports the "`DistilBERT <https://arxiv.org/abs/1910.01108>`_" model:

+-----------------------------------------+----------------------+
|                                         | distilbert_6_768_12  |
+=========================================+======================+
| distil_book_corpus_wiki_en_uncased      | ✓                    |
+-----------------------------------------+----------------------+

.. code-block:: python

    import gluonnlp as nlp; import mxnet as mx;
    model, vocab = nlp.model.get_model('distilbert_6_768_12', dataset_name='distil_book_corpus_wiki_en_uncased');
    tokenizer = nlp.data.BERTTokenizer(vocab, lower=True);
    transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=512, pair=False, pad=False);
    sample = transform(['Hello world!']);
    words, valid_len = mx.nd.array([sample[0]]), mx.nd.array([sample[1]])
    seq_encoding, cls_encoding = model(words, valid_len);

Finally, GluonNLP also suports Korean BERT pre-trained model, "`KoBERT <https://github.com/SKTBrain/KoBERT>`_", using Korean wiki dataset (`kobert_news_wiki_ko_cased`).

.. code-block:: python

    import gluonnlp as nlp; import mxnet as mx;
    model, vocab = nlp.model.get_model('bert_12_768_12', dataset_name='kobert_news_wiki_ko_cased',use_decoder=False, use_classifier=False)
    tok = nlp.data.get_tokenizer('bert_12_768_12', 'kobert_news_wiki_ko_cased')
    tok('안녕하세요.')

.. hint::

   The pre-training, fine-tuning and export scripts are available `here. </_downloads/bert.zip>`__


Sentence Classification
~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example script to fine-tune sentence classification with pre-trained
BERT model.

To enable mixed precision training with float16, set `--dtype` argument to `float16`.

Results using `bert_12_768_12`:

.. editing URL for the following table: https://tinyurl.com/y4n8q84w

+-----------------+---------------------+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
|Task Name        |Metrics              |Results on Dev Set     |log                                                                                                                                         |command                                                                                                                                                          |
+=================+=====================+=======================+============================================================================================================================================+=================================================================================================================================================================+
| CoLA            |Matthew Corr.        |60.32                  |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_CoLA_base_mx1.6.0rc1.log>`__                                 |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_CoLA_base_mx1.6.0rc1.sh>`__                                                   |
+-----------------+---------------------+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| SST-2           |Accuracy             |93.46                  |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_SST_base_mx1.6.0rc1.log>`__                                  |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_SST_base_mx1.6.0rc1.sh>`__                                                    |
+-----------------+---------------------+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MRPC            |Accuracy/F1          |88.73/91.96            |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_MRPC_base_mx1.6.0rc1.log>`__                                 |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_MRPC_base_mx1.6.0rc1.sh>`__                                                   |
+-----------------+---------------------+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| STS-B           |Pearson Corr.        |90.34                  |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_STS-B_base_mx1.6.0rc1.log>`__                                |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_STS-B_base_mx1.6.0rc1.sh>`__                                                  |
+-----------------+---------------------+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| QQP             |Accuracy             |91                     |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_QQP_base_mx1.6.0rc1.log>`__                                  |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_QQP_base_mx1.6.0rc1.sh>`__                                                    |
+-----------------+---------------------+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MNLI            |Accuracy(m/mm)       |84.29/85.07            |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_MNLI_base_mx1.6.0rc1.log>`__                                 |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_MNLI_base_mx1.6.0rc1.sh>`__                                                   |
+-----------------+---------------------+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| XNLI (Chinese)  |Accuracy             |78.43                  |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_XNLI_base_mx1.6.0rc1.log>`__                                 |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_XNLI-B_base_mx1.6.0rc1.sh>`__                                                 |
+-----------------+---------------------+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+
| RTE             |Accuracy             |74                     |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_RTE_base_mx1.6.0rc1.log>`__                                  |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_RTE_base_mx1.6.0rc1.sh>`__                                                    |
+-----------------+---------------------+-----------------------+--------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+



Results using `roberta_12_768_12`:

.. editing URL for the following table: https://www.shorturl.at/cjAO7

+---------------------+------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| Dataset             | SST-2                                                                                                | MNLI-M/MM                                                                                                        |
+=====================+======================================================================================================+==================================================================================================================+
| Validation Accuracy | 95.3%                                                                                                | 87.69%, 87.23%                                                                                                   |
+---------------------+------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| Log                 | `log  <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/roberta/finetuned_sst.log>`__      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/roberta/mnli_1e-5-32.log>`__          |
+---------------------+------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| Command             | `command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/roberta/finetuned_sst.sh>`__    | `command  <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/roberta/finetuned_mnli.sh>`__    |
+---------------------+------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+

.. editing URL for the following table: https://tinyurl.com/y5rrowj3

Question Answering on SQuAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Dataset   | SQuAD 1.1                                                                                                                               | SQuAD 1.1                                                                                                                                | SQuAD 2.0                                                                                                                                |
+===========+=========================================================================================================================================+==========================================================================================================================================+==========================================================================================================================================+
| Model     | bert_12_768_12                                                                                                                          | bert_24_1024_16                                                                                                                          | bert_24_1024_16                                                                                                                          |
+-----------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| F1 / EM   | 88.58 / 81.26                                                                                                                           | 90.97 / 84.22                                                                                                                            | 81.27 / 78.14                                                                                                                            |
+-----------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Log       | `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_squad1.1_base_mx1.6.0rc1.log>`__                         | `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_squad1.1_large_mx1.6.0rc1.log>`__                         | `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_squad2.0_large_mx1.6.0rc1.log>`__                         |
+-----------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Command   | `command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_squad1.1_base_mx1.6.0rc1.sh>`__                      | `command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_squad1.1_large_mx1.6.0rc1.sh>`__                      | `command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_squad2.0_large_mx1.6.0rc1.sh>`__                      |
+-----------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Prediction| `predictions.json <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_squad1.1_base_mx1.6.0rc1.json>`__           | `predictions.json <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_squad1.1_large_mx1.6.0rc1.json>`__           | `predictions.json <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetune_squad2.0_large_mx1.6.0rc1.json>`__           |
+-----------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+

For all model settings above, we set learing rate = 3e-5 and optimizer = adam.

Note that the BERT model is memory-consuming. If you have limited GPU memory, you can use the following command to accumulate gradient to achieve the same result with a large batch size by setting *accumulate* and *batch_size* arguments accordingly.

.. code-block:: console

    $ python finetune_squad.py --optimizer adam --accumulate 2 --batch_size 6 --lr 3e-5 --epochs 2 --gpu

We support multi-GPU training via horovod:

.. code-block:: console

    $ HOROVOD_WITH_MXNET=1 HOROVOD_GPU_ALLREDUCE=NCCL pip install horovod --user --no-cache-dir
    $ horovodrun -np 8 python finetune_squad.py --bert_model bert_24_1024_16 --batch_size 4 --lr 3e-5 --epochs 2 --gpu --dtype float16 --comm_backend horovod

SQuAD 2.0
+++++++++

For SQuAD 2.0, you need to specify the parameter *version_2* and specify the parameter *null_score_diff_threshold*. Typical values are between -1.0 and -5.0. Use the following command to fine-tune the BERT large model on SQuAD 2.0 and generate predictions.json.

To get the score of the dev data, you need to download the dev dataset (`dev-v2.0.json <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json>`_) and the evaluate script (`evaluate-2.0.py <https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/>`_). Then use the following command to get the score of the dev dataset.

.. code-block:: console

    $ python evaluate-v2.0.py dev-v2.0.json predictions.json

BERT INT8 Quantization
~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example scripts to quantize fine-tuned
BERT models into int8 data type. Note that INT8 Quantization needs a nightly
version of `mxnet-mkl <https://apache-mxnet.s3-us-west-2.amazonaws.com/dist/index.html>`_.

Sentence Classification
+++++++++++++++++++++++

+-----------+-------------------+---------------+---------------+---------+---------+------------------------------------------------------------------------------------------------------------------------+
|  Dataset  | Model             | FP32 Accuracy | INT8 Accuracy | FP32 F1 | INT8 F1 | Command                                                                                                                |
+===========+===================+===============+===============+=========+=========+========================================================================================================================+
| MRPC      | bert_12_768_12    | 87.01         | 87.01         | 90.97   | 90.88   |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/calibration_MRPC_base_mx1.6.0b20200125.sh>`__ |
+-----------+-------------------+---------------+---------------+---------+---------+------------------------------------------------------------------------------------------------------------------------+
| SST-2     | bert_12_768_12    | 93.23         | 93.00         |         |         |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/calibration_SST_base_mx1.6.0b20200125.sh>`__  |
+-----------+-------------------+---------------+---------------+---------+---------+------------------------------------------------------------------------------------------------------------------------+

Question Answering
++++++++++++++++++

+-----------+-------------------+---------+---------+---------+---------+----------------------------------------------------------------------------------------------------------------------------+
|  Dataset  | Model             | FP32 EM | INT8 EM | FP32 F1 | INT8 F1 | Command                                                                                                                    |
+===========+===================+=========+=========+=========+=========+============================================================================================================================+
| SQuAD 1.1 | bert_12_768_12    | 81.18   | 80.32   | 88.58   | 88.10   |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/calibration_squad1.1_base_mx1.6.0b20200125.sh>`__ |
+-----------+-------------------+---------+---------+---------+---------+----------------------------------------------------------------------------------------------------------------------------+

For all model settings above, we use a subset of evaluation dataset for calibration.

Pre-training from Scratch
~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide scripts for pre-training BERT with masked language modeling and and next sentence prediction.

The pre-training data format expects: (1) One sentence per line. These should ideally be actual sentences, not entire paragraphs or arbitrary spans of text for the "next sentence prediction" task. (2) Blank lines between documents. You can find a sample pre-training text with 3 documents `here <https://github.com/dmlc/gluon-nlp/blob/master/scripts/bert/sample_text.txt>`__. You can perform sentence segmentation with an off-the-shelf NLP toolkit such as NLTK.


.. hint::

   You can download pre-processed English wikipedia dataset `here. <https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/enwiki-197b5d8d.zip>`__


Pre-requisite
+++++++++++++

We recommend horovod for scalable multi-gpu multi-machine training.

To install horovod, you need:

- `NCCL <https://developer.nvidia.com/nccl>`__, and
- `OpenMPI <https://www.open-mpi.org/software/ompi/v4.0/>`__

Then you can install horovod via the following command:

.. code-block:: console

    $ HOROVOD_WITH_MXNET=1 HOROVOD_GPU_ALLREDUCE=NCCL pip install horovod==0.16.2 --user --no-cache-dir

Run Pre-training
++++++++++++++++

You can use the following command to run pre-training with 2 hosts, 8 GPUs each:

.. code-block:: console

    $ mpirun -np 16 -H host0_ip:8,host1_ip:8 -mca pml ob1 -mca btl ^openib \
             -mca btl_tcp_if_exclude docker0,lo --map-by ppr:4:socket \
             --mca plm_rsh_agent 'ssh -q -o StrictHostKeyChecking=no' \
             -x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
             -x MXNET_SAFE_ACCUMULATION=1 --tag-output \
             python run_pretraining.py --data='folder1/*.txt,folder2/*.txt,' \
             --data_eval='dev_folder/*.txt,' --num_steps 1000000 \
             --lr 1e-4 --total_batch_size 256 --accumulate 1 --raw --comm_backend horovod

If you see out-of-memory error, try increasing --accumulate for gradient accumulation.

When multiple hosts are present, please make sure you can ssh to these nodes without password.

Alternatively, if horovod is not available, you could run pre-training with the MXNet native parameter server by setting --comm_backend and --gpus.

.. code-block:: console

    $ MXNET_SAFE_ACCUMULATION=1 python run_pretraining.py --comm_backend device --gpus 0,1,2,3,4,5,6,7 ...

The BERT base model produced by gluonnlp pre-training script (`log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/bert_base_pretrain.log>`__) achieves 83.6% on MNLI-mm, 93% on SST-2, 87.99% on MRPC and 80.99/88.60 on SQuAD 1.1 validation set on the books corpus and English wikipedia dataset.

Custom Vocabulary
+++++++++++++++++

The pre-training script supports subword tokenization with a custom vocabulary using `sentencepiece <https://github.com/google/sentencepiece>`__.

To install sentencepiece, run:

.. code-block:: console

    $ pip install sentencepiece==0.1.82 --user

You can `train <//github.com/google/sentencepiece/tree/v0.1.82/python#model-training>`__ a custom sentencepiece vocabulary by specifying the vocabulary size:

.. code-block:: python

    import sentencepiece as spm
    spm.SentencePieceTrainer.Train('--input=a.txt,b.txt --unk_id=0 --pad_id=3 --model_prefix=my_vocab --vocab_size=30000 --model_type=BPE')

To use sentencepiece vocab for pre-training, please set --sentencepiece=my_vocab.model when using run_pretraining.py.



Export BERT for Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~

The script deploy.py allows exporting/importing BERT models. Supported values for --task argument include QA (question-answering), embedding (see below section), and classification and regression tasks specifying one of the following datasets: MRPC, QQP, QNLI, RTE, STS-B, CoLA, MNLI, WNLI, SST, XNLI, LCQMC, ChnSentiCorp. It uses available validation datasets to perform and test inference.

.. code-block:: console

    $ MXNET_SAFE_ACCUMULATION=1 MXNET_FC_TRUE_FP16=1 python deploy.py --task SST --model_parameters /path/to/saved/ckpt.params --output_dir /path/to/output/dir/ --seq_length 128 --gpu 0 --dtype float16

This will export the BERT model and its parameters for a classification (sentiment analysis) task to symbol.json/param files, saved into the directory specified by --output_dir.

Once the model is exported, you can import the model by setting --only_infer, and specifying the path to your model with --exported_model followed by the prefix name of the symbol.json/param files.

The batch size can be specified via --test_batch_size option, and accuracy can be checked setting --check_accuracy.

When using GPU and data type FP16 (--dtype float16), we recommend to use MXNET_FC_TRUE_FP16=1 for boosting performance.
Moreover, you can use a custom graph pass for BERT, via --custom_pass [custom_pass_file], to improve the performance on GPU. To generate the pass you can run setup.py within the BERT scripts directory. These GPU optimizations require MXNet version 1.7 or higher.


BERT for Sentence or Tokens Embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goal of this BERT Embedding is to obtain the token embedding from BERT's pre-trained model. In this way, instead of building and do fine-tuning for an end-to-end NLP model, you can build your model by just utilizing the token embeddings. You can use the command line interface below:

.. code-block:: shell

    python embedding.py --sentences "GluonNLP is a toolkit that enables easy text preprocessing, datasets loading and neural models building to help you speed up your Natural Language Processing (NLP) research."
    Text: g ##lu ##on ##nl ##p is a tool ##kit that enables easy text prep ##ro ##ces ##sing , data ##set ##s loading and neural models building to help you speed up your natural language processing ( nl ##p ) research .
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
