
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

Sentiment Analysis through textCNN Model
----------------------------------------------------

:download:`[Download] </scripts/sentiment_analysis.zip>`

This script can be used to train a sentiment analysis model.
The convolutional models is loaded from Gluon NLP Toolkit model zoo. It also showcases how to use different 
bucketing strategies to speed up training.

epoch:
|    |  MR  |  SST-1  |  SST-2  | Subj   |  TREC  |
| --- | --- | --- | --- | --- | --- |
|  rand  |  todo  |  todo  |   todo |  todo  |  todo  |
|  static  |  todo  |   todo |  todo  |   todo |  todo  |
|  non-static  |   todo |  todo  |   todo | todo   | todo   |
|  multichannel  |  todo  |  todo  |   todo |   todo |  todo  |

log:
|    |  MR  |  SST-1  |  SST-2  | Subj   |  TREC  |
| --- | --- | --- | --- | --- | --- |
|  rand  |  [1]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_mr_rand.log>` |  [5]  /`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_sst1_rand.log>`|   [9] /`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_sst2_rand.log>`|  [13]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_subj_rand.log>`  |  [17]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_trec_rand.log>`  |
|  static  |  [2]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_mr_static.log>`  |   [6]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_sst1_static.log>` |  [10]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_sst2_static.log>` |   [14]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_subj_static.log>` |  [18]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_trec_static.log>`  |
|  non-static  |   [3]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_mr_non-static.log>` |  [7]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_sst1_non-static.log>`   |   [11]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_sst2_non-static.log>`  | [15]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_subj_non-static.log>`    | [19]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_trec_non-static.log>`    |
|  multichannel  |  [4]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_mr_multichannel.log>`   |  [8]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_sst1_multichannel.log>`  |   [12]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_sst2_multichannel.log>` |   [16]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_subj_multichannel.log>` |  [20]/`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/sentiment_analysis/cache_standard_trec_multichannel.log>`  |

acc:
|    |  MR  |  SST-1  |  SST-2  | Subj   |  TREC  |
| --- | --- | --- | --- | --- | --- |
|  rand  |  todo  |  todo  |   todo |  todo  |  todo  |
|  static  |  todo  |   todo |  todo  |   todo |  todo  |
|  non-static  |   todo |  todo  |   todo | todo   | todo   |
|  multichannel  |  todo  |  todo  |   todo |   todo |  todo  |

[1]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 20 --dropout 0.5 --lr 0.005 --valid_ratio 0.1 --save-prefix sa_cnn_300 --model_mode multichannel --data_name MR

[2]:

.. code-block:: console

   $ python sentiment_analysis_cnn.py --gpu 0 --batch_size 50 --epochs 20 --dropout 0.5 --lr 0.005 --valid_ratio 0.1 --save-prefix sa_cnn_300 --model_mode multichannel --data_name MR

...

[1] Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.
