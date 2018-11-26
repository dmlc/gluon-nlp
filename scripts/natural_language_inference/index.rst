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

Train the model without intra-sentence attention:

.. code-block:: console

	$ python3 main.py --train-file data/snli_1.0/train.txt --test-file data/snli_1.0/dev.txt --output-dir output/snli-basic --batch-size 32 --print-interval 5000 --lr 0.025 --epochs 300 --gpu-id 0 --dropout 0.2 --weight-decay 1e-5

Test:

.. code-block:: console

	$ python3 main.py --test-file data/snli_1.0/test.txt --model-dir output/snli-basic --gpu-id 0 --mode test --output-dir output/snli-basic/test

We achieve 84.6% accuracy on the SNLI test set, comparable to 86.3% reported in the
original paper.

Train the model with intra-sentence attention:

.. code-block:: console

	$ python3 main.py --train-file data/snli_1.0/train.txt --test-file data/snli_1.0/dev.txt --output-dir output/snli-intra --batch-size 32 --print-interval 5000 --lr 0.025 --epochs 300 --gpu-id 0 --dropout 0.2 --weight-decay 1e-5 --intra-attention

Test:

.. code-block:: console

	$ python3 main.py --test-file data/snli_1.0/test.txt --model-dir output/snli-intra --gpu-id 0 --mode test --output-dir output/snli-intra/test

We achieve 84.4% accuracy on the SNLI test set, compared to 86.8% reported in the
original paper.
Note that our intra-sentence attention implementation omitted the
distance-sensitive bias term described in Equation (7) in the original paper.

