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

+----------------+--------+---------+---------+--------+--------+--------+--------+
|                | MR     | SST-1   | SST-2   | Subj   | TREC   |   CR   |  MPQA  |
+================+========+=========+=========+========+========+========+========+
| rand           |   200  |   200   |   200   |   200  |   200  |   200  |   200  |
+----------------+--------+---------+---------+--------+--------+--------+--------+
| static         |   200  |   200   |   200   |   200  |   200  |   200  |   200  |
+----------------+--------+---------+---------+--------+--------+--------+--------+
| non-static     |   200  |   200   |   200   |   200  |   200  |   200  |   200  |
+----------------+--------+---------+---------+--------+--------+--------+--------+
| multichannel   |   200  |   200   |   200   |   200  |   200  |   200  |   200  |
+----------------+--------+---------+---------+--------+--------+--------+--------+

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


test accuracy (SST-1, SST-2, and TREC) or cross-validation accuracy (MR, Subj, CR and MPQA):

+----------------+----------+-----------+-----------+----------+----------+----------+----------+
|                |   MR     |   SST-1   |   SST-2   |   Subj   |   TREC   |    CR    |   MPQA   |
+================+==========+===========+===========+==========+==========+==========+==========+
| rand           |   76.0   |   44.9    |   82.0    |   89.4   |   90.0   |   79.1   |   85.4   |
+----------------+----------+-----------+-----------+----------+----------+----------+----------+
| static         |   79.9   |   46.6    |   86.9    |   92.0   |   91.4   |   82.9   |   89.4   |
+----------------+----------+-----------+-----------+----------+----------+----------+----------+
| non-static     |   80.2   |   46.1    |   86.5    |   92.4   |   92.4   |   83.4   |   89.4   |
+----------------+----------+-----------+-----------+----------+----------+----------+----------+
| multichannel   |   80.2   |   47.9    |   86.0    |   92.9   |   92.8   |   83.6   |   89.8   |
+----------------+----------+-----------+-----------+----------+----------+----------+----------+

[1]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode rand --data_name MR

[2]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode static --data_name MR


[3]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode non-static --data_name MR


[4]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode multichannel --data_name MR

[5]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode rand --data_name SST-1

[6]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode static --data_name SST-1

[7]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode non-static --data_name SST-1

[8]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode multichannel --data_name SST-1

[9]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode rand --data_name SST-2

[10]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode static --data_name SST-2

[11]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode non-static --data_name SST-2

[12]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode multichannel --data_name SST-2

[13]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode rand --data_name Subj

[14]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode static --data_name Subj

[15]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode non-static --data_name Subj

[16]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode multichannel --data_name Subj

[17]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode rand --data_name TREC

[18]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode static --data_name TREC

[19]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode non-static --data_name TREC

[20]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode multichannel --data_name TREC
   
[21]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode rand --data_name CR

[22]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode static --data_name CR

[23]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode non-static --data_name CR

[24]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode multichannel --data_name CR
   
[25]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode rand --data_name MPQA

[26]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode static --data_name MPQA

[27]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode non-static --data_name MPQA

[28]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 200 --dropout 0.5 --model_mode multichannel --data_name MPQA

