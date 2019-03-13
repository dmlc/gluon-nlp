Example Usage of Exporting Hybridizable BERT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For exporting hybridizable BERT, we mean the BERT with fixed input embedding size and sequence length can be exported through
a static shape based implementation of hybridblock based BERT. By using this, we can export a block based BERT model.

Please follow the steps below for exporting the model.

Step 1: Load dataset. An example can be found in 'static_export_squad.py'.


Step 2: create a hybridizable task guided model using BERT:

.. code-block:: console

    $ class StaticBertForQA(HybridBlock)

An example can be found in 'static_bert_for_qa_model.py'.


Step 3: specify the static input size and sequence length of the input data via environment variables
in the training script:

.. code-block:: console

    $ os.environ['SEQLENGTH'] = str(args.seq_length)
    $ os.environ['INPUTSIZE'] = str(args.input_size)

An example can be found in 'static_export_squad.py'.


Step 4: hybridize the model in the script:

.. code-block:: console

    $ net = StaticBertForQA(bert=bert)
    $ net.hybridize(static_alloc=True, static_shape=True)

An example can be found in 'static_export_squad.py'.


Step 5: export trained model:

.. code-block:: console

    $ net.export(os.path.join(args.output_dir, 'static_net'), epoch=args.epochs)

To export the model, in 'static_export_squad.py', set export=True.


Example Usage of Finetuning Hybridizable BERT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example mainly introduces the steps needed to use the hybridizable BERT models to finetune on a specific NLP task.
We use SQuAD dataset for Question Answering as an example.

Step 1: create a hybridizable task guided model using BERT:

.. code-block:: console

    $ class StaticBertForQA(HybridBlock)

An example can be found in 'static_bert_for_qa_model.py'.


Step 2: specify the input size and sequence length of the input data via environment variables
in the training script:

.. code-block:: console

    $ os.environ['SEQLENGTH'] = str(args.seq_length)
    $ os.environ['INPUTSIZE'] = str(args.input_size)

An example can be found in 'static_finetune_squad.py'.


Step 3: hybridize the model in the training script:

.. code-block:: console

    $ net = StaticBertForQA(bert=bert)
    $ net.hybridize(static_alloc=True, static_shape=True)

An example can be found in 'static_finetune_squad.py'.


Step 4: export trained model:

.. code-block:: console

    $ net.export(os.path.join(args.output_dir, 'static_net'), epoch=args.epochs)

To export the model, in 'static_finetune_squad.py', set export=True.


For all model settings above, we set learning rate = 3e-5 and optimizer = adam.


+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
|                       | SQuAD 1.1                                                                                                                  | SQuAD 1.1                                                                                                                   | SQuAD 2.0                                                                                                                   |
+=======================+============================================================================================================================+=============================================================================================================================+=============================================================================================================================+
| model                 | bert_12_768_12                                                                                                             | bert_24_1024_16                                                                                                             | bert_24_1024_16                                                                                                             |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| F1                    | 88.54                                                                                                                      | 90.84                                                                                                                       | 81.46                                                                                                                       |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| EM                    | 81.10                                                                                                                      | 84.03                                                                                                                       | 78.49                                                                                                                       |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| batch_size            | 12                                                                                                                         | 4                                                                                                                           | 4                                                                                                                           |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| gradient accumulation | None                                                                                                                       | 6                                                                                                                           | 8                                                                                                                           |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| epochs                | 2                                                                                                                          | 2                                                                                                                           | 2                                                                                                                           |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| training log          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/static_finetune_squad1.1_base.log>`__      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/static_finetune_squad1.1_large.log>`__      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluonnlp/logs/bert/static_finetune_squad2.0_large.log>`__      |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| command               | [8]                                                                                                                        | [9]                                                                                                                         | [10]                                                                                                                        |
+-----------------------+----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+

BERT BASE on SQuAD 1.1
++++++++++++++++++++++

[1] bert_12_768_12

.. code-block:: console

    $ python static_finetune_squad.py --optimizer adam --batch_size 12 --lr 3e-5 --epochs 2 --gpu 0 --export


BERT LARGE on SQuAD 1.1
+++++++++++++++++++++++

[2] bert_24_1024_16

.. code-block:: console

    $ python static_finetune_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 6 --batch_size 4 --lr 3e-5 --epochs 2 --gpu 0 --export


BERT LARGE on SQuAD 2.0
+++++++++++++++++++++++

[3] bert_24_1024_16

.. code-block:: console

    $ python static_finetune_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 8 --batch_size 4 --lr 3e-5 --epochs 2 --gpu 0 --null_score_diff_threshold -2.0 --version_2 --export

To get the score of the dev data, you need to download the dev dataset (`dev-v2.0.json <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json>`_) and the evaluate script (`evaluate-2.0.py <https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/>`_). Then use the following command to get the score of the dev dataset.

.. code-block:: console

    $ python evaluate-v2.0.py dev-v2.0.json predictions.json

.. code-block:: json

    {
        "exact": 78.49743114629833,
        "f1": 81.46366127573552,
        "total": 11873,
        "HasAns_exact": 73.38056680161944,
        "HasAns_f1": 79.32153345593925,
        "HasAns_total": 5928,
        "NoAns_exact": 83.59966358284272,
        "NoAns_f1": 83.59966358284272,
        "NoAns_total": 5945
    }