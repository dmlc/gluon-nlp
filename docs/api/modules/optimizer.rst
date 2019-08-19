gluonnlp.optimizer
======================

Gluonnlp provides some special optimizers for training in natural language processing.

.. currentmodule:: gluonnlp.optimizer

BERTAdam Optimizer
--------------------------

The Adam optimizer with weight decay regularization for BERT.

.. autosummary::
    :nosignatures:

    BERTAdam

LAMB Optimizer
--------------------------

Implementation of the LAMB optimizer from the paper `Reducing BERT Pre-Training Time from 3 Days to 76 Minutes. <https://arxiv.org/abs/1904.00962>`_ 

In paper, the empirical results demonstrate the superior performance of LAMB for BERT and ResNet-50 training.
By increasing the batch size to the memory limit of a TPUv3 pod, BERT training time can be reduced from 3 days to 76 minutes.

.. code-block:: none

    @inproceedings{You2019LargeBO,
      title={Large Batch Optimization for Deep Learning: Training BERT in 76 minutes},
      author={Yang You and Jing Li and Sashank J. Reddi and Jonathan Hseu and Sanjiv Kumar and Srinadh Bhojanapalli and Xiaodan Song and James Demmel and Cho-Jui Hsieh},
      year={2019}}

.. autosummary::
    :nosignatures:

    LAMB

API Reference
-------------

.. automodule:: gluonnlp.optimizer
   :members:
   :imported-members:
