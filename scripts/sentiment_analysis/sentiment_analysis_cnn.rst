Sentiment Analysis through textCNN model
----------------------------------------------------

:download:`[Download] </scripts/sentiment_analysis.zip>`

This script can be used to train a sentiment analysis model.
The convolutional models is loaded from Gluon NLP Toolkit model zoo. It also showcases how to use different 
bucketing strategies to speed up training.

Use the following command to run the model

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 3 --dropout 0.5 --lr 0.005 --valid_ratio 0.1 --save-prefix imdb_cnn_300 --model_mode multichannel --data_name MR
