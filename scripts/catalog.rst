Model Catalog
=============




Language Model
--------------
`Language Model Model Zoo Index <./language_model/index.html>`_

Word Language Model
~~~~~~~~~~~~~~~~~~~

Dataset: Wikitext-2

+---------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| Pre-trained Model                     | Test Perplexity |Training Command                                                                                                             | log                                                                                                                         |
+=======================================+=================+=============================================================================================================================+=============================================================================================================================+
| standard_lstm_lm_200_wikitext-2  [1]_ | 101.64          |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/standard_lstm_lm_200_wikitext-2.sh>`__   |  `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/standard_lstm_lm_200_wikitext-2.log>`__    |
+---------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| standard_lstm_lm_650_wikitext-2  [1]_ | 86.91           |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/standard_lstm_lm_650_wikitext-2.sh>`__   |  `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/standard_lstm_lm_650_wikitext-2.log>`__    |
+---------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| standard_lstm_lm_1500_wikitext-2 [1]_ | 82.29           |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/standard_lstm_lm_1500_wikitext-2.sh>`__  |  `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/standard_lstm_lm_1500_wikitext-2.log>`__   |
+---------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| awd_lstm_lm_600_wikitext-2       [1]_ | 80.67           |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/awd_lstm_lm_600_wikitext-2.sh>`__        |  `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/awd_lstm_lm_600_wikitext-2.log>`__         |
+---------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| awd_lstm_lm_1150_wikitext-2      [1]_ | 65.62           |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/awd_lstm_lm_1150_wikitext-2.sh>`__       |  `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/awd_lstm_lm_1150_wikitext-2.log>`__        |
+---------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+


Cache Language Model
~~~~~~~~~~~~~~~~~~~~

Dataset: Wikitext-2

+---------------------------------------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| Pre-trained Model                           | Test Perplexity |Training Command                                                                                                                  | log                                                                                                                           |
+=============================================+=================+==================================================================================================================================+===============================================================================================================================+
| cache_awd_lstm_lm_1150_wikitext-2      [2]_ | 51.46           |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/cache_awd_lstm_lm_1150_wikitext-2.sh>`__      |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/cache_awd_lstm_lm_1150_wikitext-2.log>`__      |
+---------------------------------------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| cache_awd_lstm_lm_600_wikitext-2       [2]_ | 62.19           |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/cache_awd_lstm_lm_600_wikitext-2.sh>`__       |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/cache_awd_lstm_lm_600_wikitext-2.log>`__       |
+---------------------------------------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| cache_standard_lstm_lm_1500_wikitext-2 [2]_ | 62.79           |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/cache_standard_lstm_lm_1500_wikitext-2.sh>`__ |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/cache_standard_lstm_lm_1500_wikitext-2.log>`__ |
+---------------------------------------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| cache_standard_lstm_lm_650_wikitext-2  [2]_ | 65.85           |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/cache_standard_lstm_lm_650_wikitext-2.sh>`__  |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/cache_standard_lstm_lm_650_wikitext-2.log>`__  |
+---------------------------------------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| cache_standard_lstm_lm_200_wikitext-2  [2]_ | 73.74           |`command <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/cache_standard_lstm_lm_200_wikitext-2.sh>`__  |`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/language_model/cache_standard_lstm_lm_200_wikitext-2.log>`__  |
+---------------------------------------------+-----------------+----------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+




.. [1] Merity, S., et al.  \
       "`Regularizing and optimizing LSTM language models <https://openreview.net/pdf?id=SyyGPP0TZ>`_". \
       ICLR 2018
.. [2] Grave, E., et al. \
       "`Improving neural language models with a continuous cache <https://openreview.net/pdf?id=B184E5qee>`_".\
       ICLR 2017
