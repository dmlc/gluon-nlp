Word Embedding Toolkit
----------------------
:download:`[Download] </scripts/word_embeddings.zip>`

Gluon NLP makes it easy to evaluate and train word embeddings. This folder
includes examples to evaluate the pre-trained embeddings included in the Gluon
NLP toolkit as well as example scripts for training embeddings on custom
datasets.


Word Embedding Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate a specific embedding on one or multiple datasets you can use the
included `evaluate_pretrained.py` as follows.


.. code-block:: console

   $ python evaluate_pretrained.py

Call the script with the `--help` option to get an overview of the supported
options.

The download link above contains a notebook with extended results comparing the
different included pretrained embeddings on all Word Embedding Evaluation
datasets included in the toolkit, providing detailed information per category in
the respective datasets.

We include a `run_all.sh` script to reproduce the results.


.. code-block:: console

   $ run_all.sh


Loading of binary fasttext models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fasttext models trained with the library of facebookresearch are exported both
in a text and a binary format. Unlike the text format, the binary format
preserves information about subword units and consequently supports computation
of word vectors for words unknown during training (and not included in the text
format). `evaluate_fasttext_bin.py` shows how to load the binary format into a
FasttextEmbeddingModel Block provided by the Gluon NLP toolkit.

Using this Block together with the NGramHashes subword function of the Gluon NLP
toolkit it is possible to compute word vectors for unknown words as part of your
model.

Word Embedding Training
~~~~~~~~~~~~~~~~~~~~~~~

Besides loading pretrained embeddings, the Gluon NLP toolkit also makes it easy
to train embeddings.

`train_word2vec.py` shows how to facilitate the embeddings related functionality
in the Gluon NLP toolkit to train Word2Vec word embedding models. Similarly
`train_fasttext.py` shows how to train an embedding model that facilitates
subword information.

Word2Vec models were introduced by

- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation
  of word representations in vector space. ICLR Workshop , 2013.

FastText models were introudced by

- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word
  vectors with subword information. TACL, 5(), 135–146.

We report the results obtained by running the `train_fasttext.py` script with
default parameters. You can reproduce these results with runningand `python
train_fasttext.py --gpu 0` respectively. For comparison we also report the
results obtained by training FastText with the `facebookresearch/fastText
implementation <https://github.com/facebookresearch/fastText>`_. All results are
obtained by training 10 epochs on the `Text8
<http://mattmahoney.net/dc/textdata.html>`_ dataset.


======================================  ===========================  ===================
Similarity Dataset                      facebookresearch/fasttext    train_fasttext.py
======================================  ===========================  ===================
WordSim353-similarity                                  0.663724             0.718848
WordSim353-relatedness                                 0.58125              0.606002
MEN (test set)                                         0.705311             0.663146
RadinskyMTurk                                          0.665705             0.652314
RareWords                                              0.384798             0.378155
SimLex999                                              0.302956             0.283543
SimVerb3500                                            0.195971             0.189177
SemEval17Task2 (test set)                              0.550221             0.559741
BakerVerb143                                           0.419705             0.382791
YangPowersVerb130                                      0.459764             0.374102
======================================  ===========================  ===================

===========================================  ===========================  ===================
Googal Analogy Test Set Category             facebookresearch/fasttext    train_fasttext.py
===========================================  ===========================  ===================
capital-common-countries                             0.416996                  0.577075
capital-world                                        0.114721                  0.22458
currency                                             0.0103926                 0.0935335
city-in-state                                        0.0689096                 0.143494
family                                               0.247036                  0.460474
gram1-adjective-to-adverb                            0.620968                  0.609879
gram2-opposite                                       0.619458                  0.607143
gram3-comparative                                    0.73048                   0.84009
gram4-superlative                                    0.712121                  0.57041
gram5-present-participle                             0.418561                  0.565341
gram6-nationality-adjective                          0.809256                  0.847405
gram7-past-tense                                     0.153205                  0.314103
gram8-plural                                         0.850601                  0.772523
gram9-plural-verbs                                   0.772414                  0.635632
===========================================  ===========================  ===================
