Word Embedding Evaluation
-------------------------

This script can be used to evaluate pretrained word embeddings included in the
gluon NLP toolkit.

The download link below contains a notebook with extended results comparing the
different included pretrained embeddings on all Word Embedding Evaluation
datasets included in the toolkit, providing detailed information per category in
the respective datasets.

We include a `run_all.sh` script to reproduce the results.


.. code-block:: bash

   $ run_all.sh


To evaluate a specific embedding on one or multiple datasets you can use the
included `word_embedding_evaluation.py` as follows.


.. code-block:: bash

   $ python word_embedding_evaluation.py

Call the script with the `--help` option to get an overview of the supported
options.
