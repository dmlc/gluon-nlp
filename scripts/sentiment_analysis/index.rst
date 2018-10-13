Sentiment Analysis
------------------

:download:`[Download] </model_zoo/sentiment_analysis.zip>`

Through Fine-tuning Word Language Model
+++++++++++++++++++++++++++++++++++++++

This script can be used to train a sentiment analysis model from scratch, or fine-tune a pre-trained language model.
The pre-trained language models are loaded from Gluon NLP Toolkit model zoo. It also showcases how to use different
bucketing strategies to speed up training.

Use the following command to run without using pre-trained model (`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/sentiment_raw_20180817.log>`__)

.. code-block:: console

   $ python finetune_lm.py --gpu 0 --batch_size 16 --bucket_type fixed --epochs 3 --dropout 0 --no_pretrained --lr 0.005 --valid_ratio 0.1 --save-prefix imdb_lstm_200  # Test Accuracy 85.60

Use the following command to run with pre-trained model (`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/sentiment_pretrained_20180817.log>`__)

.. code-block:: console

   $ python finetune_lm.py --gpu 0 --batch_size 16 --bucket_type fixed --epochs 3 --dropout 0 --lr 0.005 --valid_ratio 0.1 --save-prefix imdb_lstm_200  # Test Accuracy 86.46
