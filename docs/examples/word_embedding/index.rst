Representation Learning
=======================

.. container:: cards

   .. card::
      :title: Using Pre-trained Word Embeddings
      :link: word_embedding.html

      Basics on how to use word embedding with vocab in GluonNLP and apply it on word similarity and
      analogy problems.

   .. card::
      :title: Word Embeddings Training and Evaluation
      :link: word_embedding_training.html

      Learn how to train fastText and word2vec embeddings on your own dataset, and determine
      embedding quality through intrinsic evaluation.

   .. card::
      :title: Extracting Sentence Features with Pre-trained ELMo
      :link: ../sentence_embedding/elmo_sentence_representation.html

      See how to use GluonNLP's model API to automatically download the pre-trained ELMo
      model from NAACL2018 best paper, and extract features with it.

   .. card::
      :title: Fine-tuning Pre-trained BERT Models
      :link: ../sentence_embedding/bert.html

      See how to use GluonNLP to fine-tune a sentence pair classification model with
      pre-trained BERT parameters.


.. toctree::
   :hidden:
   :maxdepth: 2

   word_embedding.ipynb
   word_embedding_training.ipynb
   ../sentence_embedding/elmo_sentence_representation.ipynb
   ../sentence_embedding/bert.ipynb