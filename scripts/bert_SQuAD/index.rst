BERT for SQuAD
-------------------------------------------------------

GluonNLP provides the following example script to fine-tune SQuAD with pre-trained
BERT model.

Download the SQuAD1.1 dataset:

 .. code-block:: console
 
    $ wget -P ./squad1.1 https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
    $ wget -P ./squad1.1 https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

Use the following command to fine-tune the BERT model for SQuAD1.1 dataset.

Note that this will require more than 12G of GPU memory.
 
.. code-block:: console

    $ python finetune_squad.py --train_file ./squad1.1/train-v1.1.json --predict_file ./squad1.1/dev-v1.1.json --optimizer adam --gpu

If you are using less than 12G of GPU memory, you can use the following command to achieve a similar effect. But need Mxnet>1.5.0

Note that this will require approximately no more than 8G of GPU memory. If your GPU memory is too small, you can adjust **accumulate** and **batch_size**.

.. code-block:: console

    $ python finetune_squad.py --train_file ./squad1.1/train-v1.1.json --predict_file ./squad1.1/dev-v1.1.json --optimizer bertadam --accumulate 2 --batch_size 6 --gpu


Should produce an output like this. Explain that the F1 score on the dev dataset is 88.45%

.. code-block:: console

    {'exact_match': 81.21097445600756, 'f1': 88.4551346176558}
