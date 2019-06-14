Dependency Parsing
---------------------------------

:download:`Download scripts </model_zoo/parsing.zip>`

Deep Biaffine Dependency Parser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This package contains an implementation of `Deep Biaffine Attention for Neural Dependency Parsing <https://arxiv.org/pdf/1611.01734.pdf>`_ proposed by Dozat and Manning (2016), with SOTA accuracy.

Train
""""""""""

As the Penn Treebank dataset (PTB) is proprietary, we are unable to distribute it.
If you have a legal copy, please place it in ``tests/data/biaffine/ptb``, use this `pre-processing script <https://github.com/hankcs/TreebankPreprocessing>`_ to convert it into conllx format.
The tree view of data folder should be as follows.

.. code-block:: console

	$ tree tests/data/biaffine
	tests/data/biaffine
	└── ptb
		├── dev.conllx
		├── test.conllx
		└── train.conllx

Then Run the following code to train the biaffine model.

.. code-block:: python

    parser = DepParser()
    parser.train(train_file='tests/data/biaffine/ptb/train.conllx',
                 dev_file='tests/data/biaffine/ptb/dev.conllx',
                 test_file='tests/data/biaffine/ptb/test.conllx', save_dir='tests/data/biaffine/model',
                 pretrained_embeddings=('glove', 'glove.6B.100d'))
    parser.evaluate(test_file='tests/data/biaffine/ptb/test.conllx', save_dir='tests/data/biaffine/model')


The expected UAS should be around ``96%`` (see `training log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/syntactics/biaffine-ptb-train.log>`_ and `evaluation log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/syntactics/biaffine-ptb-test.log>`_). The trained model will be saved in following folder.

.. code-block:: console

	$ tree tests/data/biaffine/model
	tests/data/biaffine/model
	├── config.pkl
	├── model.bin
	├── test.log
	├── train.log
	└── vocab.pkl

Note that the embeddings are not kept in ``model.bin``, in order to reduce file size.
Users need to keep embeddings at the same place after training.
A good practice is to place embeddings in the model folder and distribute them together.

Decode
""""""""""

Once we trained a model or downloaded a pre-trained one, we can load it and decode raw sentences.

.. code-block:: python

    parser = DepParser()
    parser.load('tests/data/biaffine/model')
    sentence = [('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'),
                ('music', 'NN'), ('?', '.')]
    print(parser.parse(sentence))


The output should be as follows.

.. code-block:: text

	1	Is	_	_	VBZ	_	4	cop	_	_
	2	this	_	_	DT	_	4	nsubj	_	_
	3	the	_	_	DT	_	4	det	_	_
	4	future	_	_	NN	_	0	root	_	_
	5	of	_	_	IN	_	4	prep	_	_
	6	chamber	_	_	NN	_	7	nn	_	_
	7	music	_	_	NN	_	5	pobj	_	_
	8	?	_	_	.	_	4	punct	_	_
