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
options. We include a `run_all.sh` script to run the evaluation for the
pretrained English Glove and fastText embeddings included in GluonNLP.

.. code-block:: console

   $ run_all.sh

The resulting logs and a notebook containing a ranking for the different
evaluation tasks are available `here
<https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/embedding_results/>`__.


Word Embedding Training
~~~~~~~~~~~~~~~~~~~~~~~

Besides loading pretrained embeddings, the Gluon NLP toolkit also makes it easy
to train embeddings.

`train_fasttext.py` shows how to use Gluon NLP to train fastText or Word2Vec
models. The script and parts of the Gluon NLP library support just-in-time
compilation with `numba <http://numba.pydata.org/>`_, which is enabled
automatically when numba is installed on the system. Please `pip
install --upgrade numba` to make sure training speed is not needlessly throttled
by Python.

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
obtained by training 5 epochs on the `Text8
<http://mattmahoney.net/dc/textdata.html>`_ dataset.

======================================  ===========================  ===================
Similarity Dataset                        facebookresearch/fasttext    train_fasttext.py
======================================  ===========================  ===================
WordSim353-similarity                                     0.65275               0.687187
WordSim353-relatedness                                    0.540742              0.612768
MEN (test set)                                            0.659031              0.679318
RadinskyMTurk                                             0.638946              0.619085
RareWords                                                 0.40731               0.398834
SimLex999                                                 0.314253              0.309361
SimVerb3500                                               0.187372              0.190025
SemEval17Task2 (test set)                                 0.535899              0.533027
BakerVerb143                                              0.419168              0.478791
YangPowersVerb130                                         0.429905              0.437008
======================================  ===========================  ===================

===========================================  ===========================  ===================
Google Analogy Dataset                        facebookresearch/fasttext    train_fasttext.py
===========================================  ===========================  ===================
capital-common-countries                              0.337945              0.405138
capital-world                                         0.0935013             0.159151
currency                                              0.0230947             0.0427252
city-in-state                                         0.039319              0.06364
family                                                0.3083                0.300395
gram1-adjective-to-adverb                             0.694556              0.699597
gram2-opposite                                        0.76601               0.713054
gram3-comparative                                     0.721471              0.750751
gram4-superlative                                     0.727273              0.574866
gram5-present-participle                              0.5625                0.407197
gram6-nationality-adjective                           0.829268              0.826141
gram7-past-tense                                      0.173718              0.194872
gram8-plural                                          0.760511              0.848348
gram9-plural-verbs                                    0.752874              0.736782
===========================================  ===========================  ===================

Loading of fastText models with subword information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fasttext models trained with the library of facebookresearch are exported both
in a text and a binary format. Unlike the text format, the binary format
preserves information about subword units and consequently supports computation
of word vectors for words unknown during training (and not included in the text
format). Besides training new fastText embeddings with Gluon NLP it is also
possible to load the binary format into a Block provided by the Gluon NLP
toolkit using `FasttextEmbeddingModel.load_fasttext_format`.
