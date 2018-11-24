Sentiment Analysis
------------------

:download:`[Download] </model_zoo/sentiment_analysis.zip>`

Through Fine-tuning Word Language Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script can be used to train a sentiment analysis model from scratch, or fine-tune a pre-trained language model.
The pre-trained language models are loaded from Gluon NLP Toolkit model zoo. It also showcases how to use different
bucketing strategies to speed up training.

Use the following command to run without using pre-trained model (`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/sentiment_raw_20180817.log>`__)

.. code-block:: console

   $ python finetune_lm.py --gpu 0 --batch_size 16 --bucket_type fixed --epochs 3 --dropout 0 --no_pretrained --lr 0.005 --valid_ratio 0.1 --save-prefix imdb_lstm_200  # Test Accuracy 85.60

Use the following command to run with pre-trained model (`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/sentiment_pretrained_20180817.log>`__)

.. code-block:: console

   $ python finetune_lm.py --gpu 0 --batch_size 16 --bucket_type fixed --epochs 3 --dropout 0 --lr 0.005 --valid_ratio 0.1 --save-prefix imdb_lstm_200  # Test Accuracy 86.46


TextCNN
~~~~~~~


This script can be used to train a sentiment analysis model with convolutional neural networks, i.e., textCNN:

Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.

epoch:

+----------------+--------+---------+---------+--------+--------+
|                | MR     | SST-1   | SST-2   | Subj   | TREC   |
+================+========+=========+=========+========+========+
| rand           |   60   |   40    |   40    |   40   |   40   |
+----------------+--------+---------+---------+--------+--------+
| static         |   60   |   40    |   40    |   40   |   40   |
+----------------+--------+---------+---------+--------+--------+
| non-static     |   60   |   40    |   40    |   40   |   40   |
+----------------+--------+---------+---------+--------+--------+
| multichannel   |   60   |   40    |   40    |   40   |   40   |
+----------------+--------+---------+---------+--------+--------+


log:


+----------------+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
|                | MR                                                                                                       | SST-1                                                                                                       | SST-2                                                                                                        | Subj                                                                                                        | TREC                                                                                                        |
+================+==========================================================================================================+=============================================================================================================+==============================================================================================================+=============================================================================================================+=============================================================================================================+
| rand           | [1]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/MR_rand.log>`__           | [5]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/SST-1_rand.log>`__           | [9]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/SST-2_rand.log>`__            | [13]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/Subj_rand.log>`__           | [17]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/TREC_rand.log>`__           |
+----------------+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| static         | [2]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/MR_static.log>`__         | [6]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/SST-1_static.log>`__         | [10]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/SST-2_static.log>`__         | [14]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/Subj_static.log>`__         | [18]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/TREC_static.log>`__         |
+----------------+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| non-static     | [3]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/MR_non-static.log>`__     | [7]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/SST-1_non-static.log>`__     | [11]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/SST-2_non-static.log>`__     | [15]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/Subj_non-static.log>`__     | [19]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/TREC_non-static.log>`__     |
+----------------+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| multichannel   | [4]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/MR_multichannel.log>`__   | [8]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/SST-1_multichannel.log>`__   | [12]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/SST-2_multichannel.log>`__   | [16]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/Subj_multichannel.log>`__   | [20]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment/TREC_multichannel.log>`__   |
+----------------+----------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------+


test accuracy (SST-1, SST-2, and TREC) or cross-validation accuracy (MR and Subj):


+----------------+----------+-----------+-----------+----------+----------+
|                |   MR     |   SST-1   |   SST-2   |   Subj   |   TREC   |
+================+==========+===========+===========+==========+==========+
| rand           | 0.7683   | 0.5412    | 0.9358    | 0.8984   | 0.9780   |
+----------------+----------+-----------+-----------+----------+----------+
| static         | 0.7927   | 0.5421    | 0.9450    | 0.9226   | 0.9840   |
+----------------+----------+-----------+-----------+----------+----------+
| non-static     | 0.7960   | 0.5534    | 0.9387    | 0.9206   | 0.9800   |
+----------------+----------+-----------+-----------+----------+----------+
| multichannel   | 0.7999   | 0.5581    | 0.9393    | 0.9249   | 0.9900   |
+----------------+----------+-----------+-----------+----------+----------+

[1]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 60 --dropout 0.5 --lr 0.0001 --model_mode rand --data_name MR

[2]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 60 --dropout 0.5 --lr 0.0001 --model_mode static --data_name MR


[3]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 60 --dropout 0.5 --lr 0.0001 --model_mode non-static --data_name MR


[4]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 60 --dropout 0.5 --lr 0.0001 --model_mode multichannel --data_name MR

[5]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode rand --data_name SST-1

[6]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode static --data_name SST-1

[7]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode non-static --data_name SST-1

[8]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode multichannel --data_name SST-1

[9]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode rand --data_name SST-2

[10]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode static --data_name SST-2

[11]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode non-static --data_name SST-2

[12]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode multichannel --data_name SST-2

[13]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode rand --data_name Subj

[14]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode static --data_name Subj

[15]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode non-static --data_name Subj

[16]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode multichannel --data_name Subj

[17]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode rand --data_name TREC

[18]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode static --data_name TREC

[19]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode non-static --data_name TREC

[20]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 40 --dropout 0.5 --lr 0.0001 --model_mode multichannel --data_name TREC

