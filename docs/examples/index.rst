Examples
========
Here are the examples of using Gluon NLP Toolkit for modeling.

Word Embeddings
---------------
This example shows how to use the vocabulary, embedding API, and use publicly
available pre-trained embeddings. The second example shows how to evaluate the
different pretrained embeddings includede in the toolkit on a series of standard
datasets.

.. toctree::
   :maxdepth: 1

   word_embedding.ipynb
   word_embedding_evaluation.ipynb


Language Model
--------------
This example shows how to build a word-level language model on WikiText-2 with Gluon NLP Toolkit.
By using the existing data pipeline tools and building blocks, the process is greatly simplified.

.. toctree::
   :maxdepth: 1

   language_model.ipynb

Sentiment Analysis
------------------
This example shows how to load a language model pre-trained on wikitext-2 in Gluon NLP Toolkit model
zoo, and reuse the language model encoder for sentiment analysis on IMDB movie reviews dataset. It
also showcases how to use different bucketing strategies to speed up training.

.. toctree::
   :maxdepth: 1

   sentiment_analysis.ipynb
