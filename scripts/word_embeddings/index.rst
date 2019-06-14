Word Embedding
--------------

:download:`Download scripts </model_zoo/word_embeddings.zip>`

Gluon NLP makes it easy to evaluate and train word embeddings. Here are
examples to evaluate the pre-trained embeddings included in the Gluon
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


Word Embedding Training (Skipgram and CBOW)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides loading pre-trained embeddings, the Gluon NLP toolkit also makes it easy
to train embeddings.

The following code block shows how to use Gluon NLP to train a SkipGram or CBOW
models. The script and parts of the Gluon NLP library support just-in-time
compilation with `numba <http://numba.pydata.org/>`_, which is enabled
automatically when numba is installed on the system. Please `pip
install --upgrade numba` to make sure training speed is not needlessly throttled
by Python.

.. code-block:: console

   $ python train_sg_cbow.py --model skipgram --ngram-buckets 0  # Word2Vec Skipgram
   $ python train_sg_cbow.py --model skipgram --ngram-buckets 2000000  # fastText Skipgram
   $ python train_sg_cbow.py --model cbow --ngram-buckets 0  # Word2Vec CBOW
   $ python train_sg_cbow.py --model cbow --ngram-buckets 2000000  # fastText CBOW

Word2Vec models were introduced by Mikolov et al., "Efficient estimation of word
representations in vector space" ICLR Workshop 2013. FastText models were
introudced by Bojanowski et al., "Enriching word vectors with subword
information" TACL 2017.

We report the results obtained by running the :code:`python3
train_sg_cbow.py --batch-size 4096 --epochs 5 --data fil9 --model skipgram`
script.For comparison we also report the results obtained by training FastText
with the `facebookresearch/fastText implementation
<https://github.com/facebookresearch/fastText>`_. All results are obtained by
training 5 epochs on the `Fil9 <http://mattmahoney.net/dc/textdata.html>`_
dataset.

======================================  ===========================  ===================
Similarity Dataset                        facebookresearch/fastText    train_sg_cbow.py
======================================  ===========================  ===================
WordSim353-similarity                                     0.752                0.734
WordSim353-relatedness                                    0.612                0.608
MEN (test set)                                            0.736                0.700
RadinskyMTurk                                             0.687                0.655
RareWords                                                 0.420                0.457
SimLex999                                                 0.320                0.346
SimVerb3500                                               0.190                0.235
SemEval17Task2 (test set)                                 0.541                0.542
BakerVerb143                                              0.406                0.383
YangPowersVerb130                                         0.489                0.466
======================================  ===========================  ===================

===========================================  ===========================  ===================
Google Analogy Dataset                        facebookresearch/fastText    train_sg_cbow.py
===========================================  ===========================  ===================
capital-common-countries                              0.796                0.581
capital-world                                         0.442                0.334
currency                                              0.068                0.074
city-in-state                                         0.198                0.076
family                                                0.498                0.593
gram1-adjective-to-adverb                             0.377                0.688
gram2-opposite                                        0.343                0.693
gram3-comparative                                     0.646                0.868
gram4-superlative                                     0.510                0.757
gram5-present-participle                              0.445                0.792
gram6-nationality-adjective                           0.828                0.840
gram7-past-tense                                      0.385                0.380
gram8-plural                                          0.706                0.810
gram9-plural-verbs                                    0.501                0.813
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


Word Embedding Training (GloVe)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gluon NLP also supports training GloVe models.

.. code-block:: console

   $ python train_glove.py tools/build/cooccurrences.npz tools/build/vocab.txt

Where the `cooccurrences.npz` is a numpy archive containing the sparse word-word
cooccurrence matrix and vocab.txt a textfile containing the words and their
counts. They can be constructed from a text corpus using the included
`vocab_count` and `cooccur` tools. They can be used as follows

.. code-block:: console

   $ mkdir tools/build; cd tools/build; cmake ..; make
   $ ./vocab_count corpus-part1.txt corpus-part2.txt > vocab.txt
   $ ./cooccur corpus-part1.txt corpus-part2.txt < vocab.txt

Also see `./vocab_count --help` and `./cooccur --help` for configuration options
such as min-count or window-size.

GloVe models were introduced by Pennington et al., "Glove: global vectors for
word representation", ACL 2014.
