Word Language Model
-------------------

This script can be used to train language models with the given specification.

Use the following command to run the AWD language model setting

.. code-block:: bash

   $ python word_language_model.py --tied --gpus 0 --save wiki2_awdlstm.params # Val PPL 75.01 Test PPL 71.35
