Example Usage of Hybridizable BERT on SQuAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`[Download] </model_zoo/bert/staticbert.zip>`

This example mainly introduces the steps needed to use the hybridizable BERT.


Step 1: create a hybridizable task guided model using BERT:

.. code-block:: console

    $ class BertForQA(HybridBlock)

An example can be found in 'bert_for_qa.py'.

Step 2: specify the input size and sequence length of the input data via environment variables
in the training script:

.. code-block:: console

    $ os.environ['SEQLENGTH'] = str(args.seq_length)
    $ os.environ['INPUTSIZE'] = str(args.input_size)

An example can be found in 'finetune_static_squad.py'.

Step 3: hybridize the model in the training script:

.. code-block:: console

    $ net = BertForQA(bert=bert)
    $ net.hybridize(static_alloc=True, static_shape=True)

An example can be found in 'finetune_static_squad.py'.

Step 4: export trained model:

.. code-block:: console

    $ net.export('net', epoch=args.epochs)

To export the model, in 'finetune_static_squad.py', set export=True.


For all model settings above, we set learing rate = 3e-5 and optimizer = adam.

BERT BASE on SQuAD 1.1
++++++++++++++++++++++

[1] bert_12_768_12

.. code-block:: console

    $ python finetune_static_squad.py --optimizer adam --batch_size 12 --lr 3e-5 --epochs 2 --gpu 0 --export


BERT LARGE on SQuAD 1.1
+++++++++++++++++++++++

[2] bert_24_1024_16

.. code-block:: console

    $ python finetune_static_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 6 --batch_size 4 --lr 3e-5 --epochs 2 --gpu 0 --export


BERT LARGE on SQuAD 2.0
+++++++++++++++++++++++

[3] bert_24_1024_16

.. code-block:: console

    $ python finetune_static_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 8 --batch_size 4 --lr 3e-5 --epochs 2 --gpu 0 --null_score_diff_threshold -2.0 --version_2 --export

To get the score of the dev data, you need to download the dev dataset (`dev-v2.0.json <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json>`_) and the evaluate script (`evaluate-2.0.py <https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/>`_). Then use the following command to get the score of the dev dataset.

.. code-block:: console

    $ python evaluate-v2.0.py dev-v2.0.json predictions.json

.. code-block:: json

    {
        "exact": 77.958392992504,
        "f1": 81.02012658815627,
        "total": 11873,
        "HasAns_exact": 73.3974358974359,
        "HasAns_f1": 79.52968336389662,
        "HasAns_total": 5928,
        "NoAns_exact": 82.50630782169891,
        "NoAns_f1": 82.50630782169891,
        "NoAns_total": 5945
    }