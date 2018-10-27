
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


Through textCNN Model
+++++++++++++++++++++++++++++++++++++++


:download:`[Download] </scripts/sentiment_analysis.zip>`

This script can be used to train a sentiment analysis model.
The convolutional models is loaded from Gluon NLP Toolkit model zoo. It also showcases how to use different 
bucketing strategies to speed up training.

epoch:

+----------------+--------+---------+---------+--------+--------+
|                | MR     | SST-1   | SST-2   | Subj   | TREC   |
+================+========+=========+=========+========+========+
| rand           |   60   |   10    |   20    |   60   |   60   |
+----------------+--------+---------+---------+--------+--------+
| static         |   60   |   10    |   20    |   60   |   60   |
+----------------+--------+---------+---------+--------+--------+
| non-static     |   60   |   10    |   20    |   60   |   60   |
+----------------+--------+---------+---------+--------+--------+
| multichannel   |   60   |   10    |   20    |   60   |   60   |
+----------------+--------+---------+---------+--------+--------+


log:

+----------------+-------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
|                | MR                                                                                                                | SST-1                                                                                                               | SST-2                                                                                                                | Subj                                                                                                                 | TREC                                                                                                                 |
+================+===================================================================================================================+=====================================================================================================================+======================================================================================================================+======================================================================================================================+======================================================================================================================+
| rand           | [1]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/mr_rand.log>`__           | [5]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/sst1_rand.log>`__           | [9]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/sst2_rand.log>`__            | [13]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/subj_rand.log>`__           | [17]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/trec_rand.log>`__           |
+----------------+-------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| static         | [2]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/mr_static.log>`__         | [6]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/sst1_static.log>`__         | [10]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/sst2_static.log>`__         | [14]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/subj_static.log>`__         | [18]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/trec_static.log>`__         |
+----------------+-------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| non-static     | [3]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/mr_non-static.log>`__     | [7]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/sst1_non-static.log>`__     | [11]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/sst2_non-static.log>`__     | [15]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/subj_non-static.log>`__     | [19]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/trec_non-static.log>`__     |
+----------------+-------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+
| multichannel   | [4]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/mr_multichannel.log>`__   | [8]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/sst1_multichannel.log>`__   | [12]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/sst2_multichannel.log>`__   | [16]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/subj_multichannel.log>`__   | [20]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/trec_multichannel.log>`__   |
+----------------+-------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------+

acc:

+----------------+----------+-----------+-----------+----------+----------+
|                |   MR     |   SST-1   |   SST-2   |   Subj   |   TREC   |
+================+==========+===========+===========+==========+==========+
| rand           | 0.7786   | 0.5468    | 0.9358    | 0.9002   | 0.9840   |
+----------------+----------+-----------+-----------+----------+----------+
| static         | 0.7869   | 0.5558    | 0.9479    | 0.9194   | 0.9840   |
+----------------+----------+-----------+-----------+----------+----------+
| non-static     | 0.7939   | 0.5544    | 0.9404    | 0.9213   | 0.9780   |
+----------------+----------+-----------+-----------+----------+----------+
| multichannel   | 0.7953   | 0.5619    | 0.9318    | 0.9259   | 0.9880   |
+----------------+----------+-----------+-----------+----------+----------+

[1]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 20 --dropout 0.5 --lr 0.005 --valid_ratio 0.1 --save-prefix sa_cnn_300 --model_mode multichannel --data_name MR

[2]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 20 --dropout 0.5 --lr 0.005 --valid_ratio 0.1 --save-prefix sa_cnn_300 --model_mode multichannel --data_name MR

...

[1] Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.
