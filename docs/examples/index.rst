Tutorials
=========

Interested in getting started in a new NLP area? Here are some tutorials to help get started.

Data Loading and Vocabularies
-----------------------------

.. container:: cards

   .. card::
      :title: Data Loading APIs
      :link: notes/data_api.html

      Basics on how to load and process the sentiment dataset to form batches that can be processed efficiently.

   .. card::
      :title: Vocabulary APIs
      :link: notes/vocab_emb.html

      Learn how to write simple code to create index for tokens.


.. toctree::
   :hidden:
   :maxdepth: 2

   notes/index


Representation Learning
-----------------------

.. container:: cards

   .. card::
      :title: Using Pre-trained Word Embeddings
      :link: word_embedding/word_embedding.html

      Basics on how to use word embedding with vocab in GluonNLP and apply it on word similarity and
      analogy problems.

   .. card::
      :title: Word Embeddings Training and Evaluation
      :link: word_embedding/word_embedding_training.html

      Learn how to train fastText and word2vec embeddings on your own dataset, and determine
      embedding quality through intrinsic evaluation.

   .. card::
      :title: Extracting Sentence Features with Pre-trained ELMo
      :link: sentence_embedding/elmo_sentence_representation.html

      See how to use GluonNLP's API to automatically download the pre-trained ELMo model, and extract features from it.

   .. card::
      :title: Fine-tuning Pre-trained BERT Models
      :link: sentence_embedding/bert.html

      See how to use GluonNLP to fine-tune a sentence pair classification model with
      pre-trained BERT parameters.


.. toctree::
   :hidden:
   :maxdepth: 2

   word_embedding/index


Language Modeling
-----------------

.. container:: cards

   .. card::
      :title: LSTM-based Language Models
      :link: language_model/use_pretrained_lm.html

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
      :title: Training GNMT on IWSLT 2015 Dataset
      :link: machine_translation/gnmt.html

      Learn how to train Google Neural Machine Translation, a seq2seq with attention model.

   .. card::
      :title: Using Pre-trained Transformer
      :link: machine_translation/transformer.html

      Learn how to use a pre-trained transformer translation model for English-German translation.


.. toctree::
   :hidden:
   :maxdepth: 2

   machine_translation/index


Sentiment Analysis
------------------

.. container:: cards

   .. card::
      :title: Fine-tuning LSTM-based Language Model
      :link: sentiment_analysis/sentiment_analysis.html

      See how to fine-tune a pre-trained language model to perform sentiment analysis on movie reviews.

   .. card::
      :title: Training Structured Self-attentive Sentence Embedding
      :link: sentiment_analysis/self_attentive_sentence_embedding.html

      See how to use GluonNLP to build more advanced model structure for extracting sentence
      embeddings to predict Yelp review rating.


.. toctree::
   :hidden:
   :maxdepth: 2

   sentiment_analysis/index


Text Generation
---------------

.. container:: cards

   .. card::
      :title: Sequence Generation with Beam Search Sampler and Sequence Sampler
      :link: sequence_sampling/sequence_sampling.html

      Learn how to generate sentence from pre-trained language model through sampling and beam
      search.

.. toctree::
   :hidden:
   :maxdepth: 2

   sequence_sampling/index
