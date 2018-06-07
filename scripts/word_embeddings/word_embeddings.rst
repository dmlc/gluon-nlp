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
