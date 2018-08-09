entiment Analysis through Fine-tuning, w/ Bucketing
----------------------------------------------------

:download:`[Download] </scripts/sentiment_analysis.zip>`

This script can be used to train a sentiment analysis model from scratch, or fine-tune a pre-trained language model.
The pre-trained language models are loaded from Gluon NLP Toolkit model zoo. It also showcases how to use different
bucketing strategies to speed up training.

Use the following command to run without using pretrained model

.. code-block:: console

   $ python sentiment_analysis.py --gpu 0 --batch_size 16 --bucket_type fixed --epochs 3 --dropout 0 --no_pretrained --lr 0.005 --valid_ratio 0.1 --save-prefix imdb_lstm_200  # Test Accuracy 85.36

Use the following command to run with pretrained model

.. code-block:: console

   $ python sentiment_analysis.py --gpu 0 --batch_size 16 --bucket_type fixed --epochs 3 --dropout 0 --lr 0.005 --valid_ratio 0.1 --save-prefix imdb_lstm_200  # Test Accuracy 87.41

Sentiment Analysis through textCNN model
----------------------------------------------------

:download:`[Download] </scripts/sentiment_analysis.zip>`

This script can be used to train a sentiment analysis model.
The convolutional models is loaded from Gluon NLP Toolkit model zoo. It also showcases how to use different 
bucketing strategies to speed up training.

Use the following command to reproduce the textCNN paper's experimental results for CNN-multichannel on the MR dataset [1]:

.. code-block:: console

$ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 20 --dropout 0.5 --lr 0.005 --valid_ratio 0.1 --save-prefix sa_cnn_300 --model_mode multichannel --data_name MR

Use the following command to reproduce the paper's experimental results for CNN-multichannel on the SST-1 dataset [1]

.. code-block:: console

$ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 20 --dropout 0.5 --lr 0.005 --valid_ratio 0.1 --save-prefix sa_cnn_300 --model_mode multichannel --data_name SST-1

Use the following command to reproduce the textCNN paper's experimental results for CNN-multichannel on the SST-2 dataset [1]:

.. code-block:: console

$ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 20 --dropout 0.5 --lr 0.005 --valid_ratio 0.1 --save-prefix sa_cnn_300 --model_mode multichannel --data_name SST-2

Use the following command to reproduce the textCNN paper's experimental results for CNN-multichannel on the Subj dataset [1]:

.. code-block:: console

$ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 20 --dropout 0.5 --lr 0.005 --valid_ratio 0.1 --save-prefix sa_cnn_300 --model_mode multichannel --data_name Subj

Use the following command to reproduce the textCNN paper's experimental results for CNN-multichannel on the TREC dataset [1]:

.. code-block:: console

$ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 20 --dropout 0.5 --lr 0.005 --valid_ratio 0.1 --save-prefix sa_cnn_300 --model_mode multichannel --data_name TREC
