Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

BERT for SQuAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides the following example script to fine-tune SQuAD with pre-trained
BERT model.

Download the SQuAD1.1 dataset:

 .. code-block:: console
 
    $ wget -P ./squad1.1 https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
    $ wget -P ./squad1.1 https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

Use the following command to fine-tune the BERT model for SQuAD1.1 dataset.

Note that this will require more than 12G of GPU memory.
 
.. code-block:: console

    $ python finetune_squad.py --train_file ./squad1.1/train-v1.1.json --predict_file ./squad1.1/dev-v1.1.json --gpu


Should produce an output like this. Explain that the F1 score on the dev dataset is 88.4%

.. code-block:: console

    {"f1": 88.41249612335034, "exact_match": 81.2488174077578}
