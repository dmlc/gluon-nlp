Natural Language Inference
--------------------------

Replication of the model described in [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933).

Download the SNLI dataset:

.. code-block:: console

    $ mkdir data
    $ curl https://nlp.stanford.edu/projects/snli/snli_1.0.zip -o data/snli_1.0.zip
    $ unzip data/snli_1.0.zip -d data

Preprocess the data:

.. code-block:: console

	$ for split in train dev test; do python3 preprocess.py --input data/snli_1.0/snli_1.0_$split.txt --output data/snli_1.0/$split.txt; done

Train the model without intra sentence attention:

.. code-block:: console

	$ python3 main.py --train-file data/snli_1.0/train.txt --test-file data/snli_1.0/dev.txt --output-dir output/snli-basic --batch-size 32 --print-interval 5000 --lr 0.05 --epochs 300 --gpu-id 0 --dropout 0.2 --weight-decay 1e-5

Train the model with intra sentence attention:

.. code-block:: console

	$ python3 main.py --train-file data/snli_1.0/train.txt --test-file data/snli_1.0/dev.txt --output-dir output/snli-intra --batch-size 32 --print-interval 5000 --lr 0.025 --epochs 300 --gpu-id 0 --dropout 0.2 --weight-decay 1e-5 --intra-attention
