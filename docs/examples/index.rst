Tutorials
=========

Interested in getting started in a new NLP area? Here are some tutorials to help get started.


Word Embedding
--------------

.. container:: cards

   .. card::
      :title: Pre-trained Word Embeddings
      :link: word_embedding/word_embedding.html

      Basics on how to use word embedding with vocab in GluonNLP and apply it on word similarity and
      analogy problems.

   .. card::
      :title: Word Embeddings Training and Evaluation
      :link: word_embedding/word_embedding_training.html

      Learn how to train fastText and word2vec embeddings on your own dataset, and determine
      embedding quality through intrinsic evaluation.


.. toctree::
   :hidden:
   :maxdepth: 2

   word_embedding/index


Language Model
--------------

.. container:: cards

   .. card::
      :title: LSTM-based Language Models
      :link: language_model/language_model.html

      Learn what a language model is, what it can do, and how to train a word-level language model
      with truncated back-propagation-through-time (BPTT).


.. toctree::
   :hidden:
   :maxdepth: 2

   language_model/index


Machine Translation
-------------------

.. container:: cards

   .. card::
      :title: Google Neural Machine Translation
      :link: machine_translation/gnmt.html

      Learn how to train Google Neural Machine Translation, a seq2seq with attention model.

   .. card::
      :title: Machine Translation with Transformer
      :link: machine_translation/transformer.html

      Learn how to use a pre-trained transformer translation model for English-German translation.


.. toctree::
   :hidden:
   :maxdepth: 2

   machine_translation/index


Sentence Embedding
------------------

.. container:: cards

   .. card::
      :title: ELMo: Deep Contextualized Word Representations
      :link: sentence_embedding/elmo_sentence_representation.html

      See how to use GluonNLP's model API to automatically download the pre-trained ELMo
      model from NAACL2018 best paper, and extract features with it.

   .. card::
      :title: A Structured Self-attentive Sentence Embedding
      :link: sentence_embedding/self_attentive_sentence_embedding.html

      See how to use GluonNLP to build more advanced model structure for extracting sentence
      embeddings to predict Yelp review rating.

   .. card::
      :title: BERT Fine-tuning
      :link: sentence_embedding/bert.html

      See how to use GluonNLP to fine-tune a sentence pair classification model with
      pre-trained BERT parameters.

.. toctree::
   :hidden:
   :maxdepth: 2

   sentence_embedding/index


Sentiment Analysis
------------------

.. container:: cards

   .. card::
      :title: Sentiment Analysis by Fine-tuning Word Language Model
      :link: sentiment_analysis/sentiment_analysis.html

      See how to fine-tune a pre-trained language model to perform sentiment analysis on movie reviews.

.. toctree::
   :hidden:
   :maxdepth: 2

   sentiment_analysis/index


Sequence Sampling
-----------------

.. container:: cards

   .. card::
      :title: Sequence Generation with Sampling and Beam Search
      :link: sequence_sampling/sequence_sampling.html

      Learn how to generate sentence from pre-trained language model through sampling and beam
      search.

.. toctree::
   :hidden:
   :maxdepth: 2

   sequence_sampling/index
