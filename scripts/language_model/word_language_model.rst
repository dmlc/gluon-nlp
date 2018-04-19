Word Language Model
-------------------

This script can be used to train language models with the given specification.

Use the following command to run the AWDRNN language model setting

.. code-block:: bash

   $ python word_language_model.py --tied --gpus 0 --save awd_lstm_lm_1150_wikitext-2 # Val PPL 73.32 Test PPL 69.74

Use the following command to run the StandardRNN language model setting (emsize=650, nhid=650)

.. code-block:: bash

   $ python word_language_model.py --gpus 0 --emsize 650 --nhid 650 --nlayers 2 --lr 20 --epochs 750 --batch_size 20 --bptt 35 --dropout 0.5 --dropout_h 0 --dropout_i 0 --dropout_e 0 --weight_drop 0 --tied --wd 0 --alpha 0 --beta 0 --save standard_lstm_lm_650_wikitext-2 # Val PPL 98.96 Test PPL 93.90

Use the following command to run the StandardRNN language model setting (emsize=200, nhid=200)

.. code-block:: bash

   $ python word_language_model.py --gpus 0 --emsize 200 --nhid 200 --nlayers 2 --lr 20 --epochs 750 --batch_size 20 --bptt 35 --dropout 0.2 --dropout_h 0 --dropout_i 0 --dropout_e 0 --weight_drop 0 --tied --wd 0 --alpha 0 --beta 0 --save standard_lstm_lm_200_wikitext # Val PPL 108.25 Test PPL 102.26