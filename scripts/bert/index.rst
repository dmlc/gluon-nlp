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

Pre-training with BERT
~~~~~~~~~~~~~~~~~~~~~~

The scripts for masked language modeling and and next sentence prediction are also provided.

Data generation for pre-training on sample texts:

 .. code-block:: console

    $ python create_pretraining_data.py --input_file sample_text.txt --output_file out --vocab_file /home/ubuntu/.mxnet/models/book_corpus_wiki_en_uncased-c3e2bd00.vocab --do_lower_case --max_seq_length 128 --max_predictions_per_seq 20 --dupe_factor 5 --masked_lm_prob 0.15 --short_seq_prob 0.1

The data generation script takes a file path as the input (could be one or more files by wildcard). Each file contains one or more documents separated by empty lines, and each document contains one line per sentence. You can perform sentence segmentation with an off-the-shelf NLP toolkit such as NLTK.

Run pre-training with generated data:

 .. code-block:: console

    $ python run_pretraining.py --gpu --do-training --batch_size 32 --lr 2e-5 --data out.npz --warmup_ratio 0.5 --num_steps 20 --pretrained --log_interval=1 --do-eval --data_eval out.npz --batch_size_eval 8

With 20 steps of pre-training it reaches the following evaluation result on the training data::

    mlm_loss=0.082  mlm_acc=98.9  nsp_loss=0.000  nsp_acc=100.0
