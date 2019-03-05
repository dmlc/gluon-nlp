Example Usage of Hybridizable BERT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

An example can be found in 'finetune_squad.py'.

Step 3: hybridize the model in the training script:

.. code-block:: console

    $ net = BertForQA(bert=bert)
    $ net.hybridize(static_alloc=True, static_shape=True)

An example can be found in 'finetune_squad.py'.

Step 4: export trained model:

.. code-block:: console

    $ net.export('net', epoch=args.epochs)

To export the model, in 'finetune_squad.py', set export=True.

GluonNLP provides the following example script to fine-tune SQuAD with pre-trained BERT model.

The throughputs of training and inference are based on fixed sequence length=384 and input embedding size=768,
which are 1.65 samples/s and 3.97 samples/s respectively.

In total, one training epoch costs 4466.87s and inference costs 113.99s on SQuAD v1.1.

The evaluation result of the model after one training epoch is 'Exact Match': 78.78, 'F1': 86.99.

To reproduce the above result, simply run the following command with MXNet==1.5.0b20190116.
 
.. code-block:: console

    $ python finetune_squad.py --optimizer adam --gpu --export
