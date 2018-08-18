gluonnlp.model
==============

GluonNLP Toolkit supplies models for common NLP tasks with pre-trained weights. By default,
all requested pre-trained weights are downloaded from public repo and stored in ~/.mxnet/models/.

.. currentmodule:: gluonnlp.model

Language Modeling
-----------------

.. autosummary::
    :nosignatures:

    awd_lstm_lm_1150
    awd_lstm_lm_600
    AWDRNN
    standard_lstm_lm_200
    standard_lstm_lm_650
    standard_lstm_lm_1500
    big_rnn_lm_2048_512
    StandardRNN
    get_model
    BigRNN

Convolutional Encoder
----------------------

.. autosummary::
    :nosignatures:

    ConvolutionalEncoder

Highway Network
-----------------

.. autosummary::
    :nosignatures:

    Highway

Attention Cell
--------------

.. autosummary::
    :nosignatures:

    AttentionCell
    MultiHeadAttentionCell
    MLPAttentionCell
    DotProductAttentionCell

Sequence Sampling
-----------------

.. autosummary::
    :nosignatures:

    BeamSearchScorer
    BeamSearchSampler
    SequenceSampler

Other Modeling Utilities
------------------------

.. autosummary::
    :nosignatures:

    WeightDropParameter
    apply_weight_drop
    L2Normalization
    ISDense
    NCEDense
    SparseISDense
    SparseNCEDense

API Reference
-------------

.. automodule:: gluonnlp.model
    :members:
    :imported-members:
