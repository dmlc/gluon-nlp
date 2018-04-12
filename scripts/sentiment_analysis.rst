Sentiment Analysis through Fine-tuning, w/ Bucketing
----------------------------------------------------

This script can be used to train a sentiment analysis model from scratch, or fine-tune a pre-trained language model.
The pre-trained language models are loaded from Gluon NLP Toolkit model zoo. It also showcases how to use different
bucketing strategies to speed up training.

Use the following command to run without using pretrained model

.. code-block:: bash

   $ python sentiment_analysis.py --gpu 0 --batch_size 16 --bucket_type fixed --epochs 20 --dropout 0 --no_pretrained --lr 0.005 --valid_ratio 0.1 --save-prefix imdb_lstm_200  # Test Accuracy 87.88

Use the following command to run with pretrained model

.. code-block:: bash

   $ python sentiment_analysis.py --gpu 0 --batch_size 16 --bucket_type fixed --epochs 20 --dropout 0 --lr 0.005 --valid_ratio 0.1 --save-prefix imdb_lstm_200  # Test Accuracy 88.46

