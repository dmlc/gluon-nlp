Word Language Model
-------------------

This script can be used to train language models with the given specification.

Use the following command to run the small setting (embed and hidden size = 200)

.. code-block:: bash

   $ python word_language_model.py --tied --gpus 0 --save wiki2_lstm_200.params  # Test PPL 102.91

Use the following command to run the large setting (embed and hidden size = 650)

.. code-block:: bash

   $ python word_language_model.py --emsize 650 --nhid 650 --dropout 0.5 --tied --gpus 0 --save wiki2_lstm_650.params # Test PPL 89.01

