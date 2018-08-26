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
pre-trained English Glove and fastText embeddings included in GluonNLP.

.. code-block:: console

   $ run_all.sh

The resulting logs and a notebook containing a ranking for the different
evaluation tasks are available `here
<https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/embedding_results/>`__.


Word Embedding Training
~~~~~~~~~~~~~~~~~~~~~~~

Besides loading pre-trained embeddings, the Gluon NLP toolkit also makes it easy
to train embeddings.

The following code block shows how to use Gluon NLP to train fastText or Word2Vec
models. The script and parts of the Gluon NLP library support just-in-time
compilation with `numba <http://numba.pydata.org/>`_, which is enabled
automatically when numba is installed on the system. Please `pip
install --upgrade numba` to make sure training speed is not needlessly throttled
by Python.

.. code-block:: console

   $ python train_fasttext.py


Word2Vec models were introduced by

- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation
  of word representations in vector space. ICLR Workshop , 2013.

FastText models were introudced by

- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word
  vectors with subword information. TACL, 5(), 135–146.

We report the results obtained by running the :code:`train_fasttext.py` script with
default parameters. You can reproduce these results with runningand `python
train_fasttext.py --gpu 0` respectively. For comparison we also report the
results obtained by training FastText with the `facebookresearch/fastText
implementation <https://github.com/facebookresearch/fastText>`_. All results are
obtained by training 5 epochs on the `Text8
<http://mattmahoney.net/dc/textdata.html>`_ dataset.

======================================  ===========================  ===================
Similarity Dataset                        facebookresearch/fasttext    train_fasttext.py
======================================  ===========================  ===================
WordSim353-similarity                                     0.670                0.685
WordSim353-relatedness                                    0.557                0.592
MEN (test set)                                            0.665                0.629
RadinskyMTurk                                             0.640                0.609
RareWords                                                 0.400                0.429
SimLex999                                                 0.300                0.323
SimVerb3500                                               0.170                0.191
SemEval17Task2 (test set)                                 0.540                0.566
BakerVerb143                                              0.390                0.363
YangPowersVerb130                                         0.424                0.366
======================================  ===========================  ===================

===========================================  ===========================  ===================
Google Analogy Dataset                        facebookresearch/fasttext    train_fasttext.py
===========================================  ===========================  ===================
capital-common-countries                              0.581                0.470
capital-world                                         0.176                0.148
currency                                              0.046                0.043
city-in-state                                         0.100                0.076
family                                                0.375                0.342
gram1-adjective-to-adverb                             0.695                0.663
gram2-opposite                                        0.539                0.700
gram3-comparative                                     0.523                0.740
gram4-superlative                                     0.523                0.535
gram5-present-participle                              0.480                0.399
gram6-nationality-adjective                           0.830                0.830
gram7-past-tense                                      0.269                0.200
gram8-plural                                          0.703                0.860
gram9-plural-verbs                                    0.575                0.800
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
