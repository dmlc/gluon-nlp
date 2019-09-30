Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

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

.. hint::

   The pre-training, fine-tunining and export scripts are available `here. </_downloads/bert.zip>`__


Sentence Classification
~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example script to fine-tune sentence classification with pre-trained
BERT model.

To enable mixed precision training with float16, set `--dtype` argument to `float16`.

Results using `bert_12_768_12`:

.. editing URL for the following table: https://tinyurl.com/y4n8q84w

+---------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| Dataset             | MRPC                                                                                                         | RTE                                                                                                         | SST-2                                                                                                       | MNLI-M/MM                                                                                                    | XNLI (Chinese)                                                                                               |
+=====================+==============================================================================================================+=============================================================================================================+=============================================================================================================+==============================================================================================================+==============================================================================================================+
| Validation Accuracy | 88.7%                                                                                                        | 70.8%                                                                                                       | 93%                                                                                                         | 84.55%, 84.66%                                                                                               | 78.27%                                                                                                       |
+---------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| Log                 | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mrpc.log>`__       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_rte.log>`__       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_sst.log>`__       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mnli.log>`__       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_xnli.log>`__       |
+---------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| Command             | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mrpc.sh>`__    | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_rte.sh>`__    | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_sst.sh>`__    | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_mnli.sh>`__    | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetuned_xnli.sh>`__    |
+---------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+

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

+---------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Dataset | SQuAD 1.1                                                                                                                               | SQuAD 1.1                                                                                                                                | SQuAD 2.0                                                                                                                                |
+=========+=========================================================================================================================================+==========================================================================================================================================+==========================================================================================================================================+
| Model   | bert_12_768_12                                                                                                                          | bert_24_1024_16                                                                                                                          | bert_24_1024_16                                                                                                                          |
+---------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| F1 / EM | 88.53 / 80.98                                                                                                                           | 90.97 / 84.05                                                                                                                            | 77.96 / 81.02                                                                                                                            |
+---------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Log     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_base_mx1.5.0b20190216.log>`__         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_large_mx1.5.0b20190216.log>`__         | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad2.0_large_mx1.5.0b20160216.log>`__         |
+---------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Command | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_base_mx1.5.0b20190216.sh>`__      | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad1.1_large_mx1.5.0b20190216.sh>`__      | `command <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/finetune_squad2.0_large_mx1.5.0b20160216.sh>`__      |
+---------+-----------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+

For all model settings above, we set learing rate = 3e-5 and optimizer = adam.

Note that the BERT model is memory-consuming. If you have limited GPU memory, you can use the following command to accumulate gradient to achieve the same result with a large batch size by setting *accumulate* and *batch_size* arguments accordingly.

.. code-block:: console

    $ python finetune_squad.py --optimizer adam --accumulate 2 --batch_size 6 --lr 3e-5 --epochs 2 --gpu 0

SQuAD 2.0
+++++++++

For SQuAD 2.0, you need to specify the parameter *version_2* and specify the parameter *null_score_diff_threshold*. Typical values are between -1.0 and -5.0. Use the following command to fine-tune the BERT large model on SQuAD 2.0 and generate predictions.json.

To get the score of the dev data, you need to download the dev dataset (`dev-v2.0.json <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json>`_) and the evaluate script (`evaluate-2.0.py <https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/>`_). Then use the following command to get the score of the dev dataset.

.. code-block:: console

    $ python evaluate-v2.0.py dev-v2.0.json predictions.json


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


Named Entity Recognition
~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides training and prediction script for named entity recognition models.

The training script for NER requires python3 and the seqeval package:

.. code-block:: console

    $ pip3 install seqeval --user

Dataset should be formatted in `CoNLL-2003 shared task format <https://www.clips.uantwerpen.be/conll2003/ner/>`_.
Assuming data files are located in `${DATA_DIR}`, below command trains BERT model for
named entity recognition, and saves model artifacts to `${MODEL_DIR}` with `large_bert`
prefix in file names:

.. code-block:: console

    $ python3 finetune_ner.py \
        --train-path ${DATA_DIR}/train.txt \
        --dev-path ${DATA_DIR}/dev.txt \
        --test-path ${DATA_DIR}/test.txt
        --gpu 0 --learning-rate 1e-5 --dropout-prob 0.1 --num-epochs 100 --batch-size 8 \
        --optimizer bertadam --bert-model bert_24_1024_16 \
        --save-checkpoint-prefix ${MODEL_DIR}/large_bert --seed 13531

This achieves Test F1 from `91.5` to `92.2` (`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetuned_conll2003.log>`_).

Export BERT for Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Current export.py support exporting BERT models. Supported values for --task argument include classification, regression and question answering.

.. code-block:: console

    $ python export.py --task classification --model_parameters /path/to/saved/ckpt.params --output_dir /path/to/output/dir/ --seq_length 128

This will export the BERT model for classification to a symbol.json file, saved to the directory specified by --output_dir.
The --model_parameters argument is optional. If not set, the .params file saved in the output directory will be randomly initialized parameters.

BERT for Sentence or Tokens Embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goal of this BERT Embedding is to obtain the token embedding from BERT's pre-trained model. In this way, instead of building and do fine-tuning for an end-to-end NLP model, you can build your model by just utilizing the token embeddings. You can use the command line interface below:

.. code-block:: shell

    python bert/embedding.py --sentences "GluonNLP is a toolkit that enables easy text preprocessing, datasets loading and neural models building to help you speed up your Natural Language Processing (NLP) research."
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

Joint Intent Classification and Slot Labelling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Intent classification and slot labelling are two essential problems in Natural Language Understanding (NLU).
In *intent classification*, the agent needs to detect the intention that the speaker's utterance conveys. For example, when the speaker says "Book a flight from Long Beach to Seattle", the intention is to book a flight ticket.
In *slot labelling*, the agent needs to extract the semantic entities that are related to the intent. In our previous example,
"Long Beach" and "Seattle" are two semantic constituents related to the flight, i.e., the origin and the destination.

Essentially, *intent classification* can be viewed as a sequence classification problem and *slot labelling* can be viewed as a
sequence tagging problem similar to Named-entity Recognition (NER). Due to their inner correlation, these two tasks are usually
trained jointly with a multi-task objective function.

Here's one example of the ATIS dataset, it uses the `IOB2 format <https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)>`__.

+-----------+--------------------------+--------------+
| Sentence  | Tags                     | Intent Label |
+===========+==========================+==============+
| are       | O                        | atis_flight  |
+-----------+--------------------------+--------------+
| there     | O                        |              |
+-----------+--------------------------+--------------+
| any       | O                        |              |
+-----------+--------------------------+--------------+
| flight    | O                        |              |
+-----------+--------------------------+--------------+
| from      | O                        |              |
+-----------+--------------------------+--------------+
| long      | B-fromloc.city_name      |              |
+-----------+--------------------------+--------------+
| beach     | I-fromloc.city_name      |              |
+-----------+--------------------------+--------------+
| to        | O                        |              |
+-----------+--------------------------+--------------+
| columbus  | B-toloc.city_name        |              |
+-----------+--------------------------+--------------+
| on        | O                        |              |
+-----------+--------------------------+--------------+
| wednesday | B-depart_date.day_name   |              |
+-----------+--------------------------+--------------+
| april     | B-depart_date.month_name |              |
+-----------+--------------------------+--------------+
| sixteen   | B-depart_date.day_number |              |
+-----------+--------------------------+--------------+



In this example, we demonstrate how to use GluonNLP to fine-tune a pretrained BERT model for joint intent classification and slot labelling. We
choose to finetune a pretrained BERT model.  We use two datasets `ATIS <https://github.com/yvchen/JointSLU>`__ and `SNIPS <https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines>`__.

The training script requires python3 and the seqeval and tqdm packages:

.. code-block:: console

    $ pip3 install seqeval --user
    $ pip3 install tqdm --user

For the ATIS dataset, use the following command to run the experiment:

.. code-block:: console

    $ python3 finetune_icsl.py --gpu 0 --dataset atis

It produces the final slot labelling F1 = `95.83%` and intent classification accuracy = `98.66%`

For the SNIPS dataset, use the following command to run the experiment:

.. code-block:: console

    $ python3 finetune_icsl.py --gpu 0 --dataset snips

It produces the final slot labelling F1 = `96.06%` and intent classification accuracy = `98.71%`

Also, we train the models with three random seeds and report the mean/std.

For ATIS

+--------------------------------------------------------------------------------------------+----------------+-------------+
|                                             Models                                         | Intent Acc (%) | Slot F1 (%) |
+============================================================================================+================+=============+
| `Intent Gating & self-attention, EMNLP 2018 <https://www.aclweb.org/anthology/D18-1417>`__ |    98.77       |  96.52      |
+--------------------------------------------------------------------------------------------+----------------+-------------+
| `BLSTM-CRF + ELMo, AAAI 2019, <https://arxiv.org/abs/1811.05370>`__                        |    97.42       |  95.62      |
+--------------------------------------------------------------------------------------------+----------------+-------------+
| `Joint BERT, Arxiv 2019, <https://arxiv.org/pdf/1902.10909.pdf>`__                         |    97.5        |  96.1       |
+--------------------------------------------------------------------------------------------+----------------+-------------+
| Ours                                                                                       |    98.66±0.00  |  95.88±0.04 |
+--------------------------------------------------------------------------------------------+----------------+-------------+

For SNIPS

+--------------------------------------------------------------------+----------------+-------------+
|                                   Models                           | Intent Acc (%) | Slot F1 (%) |
+====================================================================+================+=============+
| `BLSTM-CRF + ELMo, AAAI 2019 <https://arxiv.org/abs/1811.05370>`__ | 99.29          | 93.90       |
+--------------------------------------------------------------------+----------------+-------------+
| `Joint BERT, Arxiv 2019 <https://arxiv.org/pdf/1902.10909.pdf>`__  | 98.60          | 97.00       |
+--------------------------------------------------------------------+----------------+-------------+
| Ours                                                               | 98.81±0.13     | 95.94±0.10  |
+--------------------------------------------------------------------+----------------+-------------+
