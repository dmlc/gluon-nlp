gluonnlp.model
==============

GluonNLP Toolkit supplies models for common NLP tasks with pre-trained weights. By default,
all requested pre-trained weights are downloaded from public repo and stored in ~/.mxnet/models/.

.. currentmodule:: gluonnlp.model

Model Registry
--------------

The model registry provides an easy interface to obtain pre-defined and pre-trained models.

.. autosummary::
    :nosignatures:

    get_model

The `get_model` function returns a pre-defined model given the name of a
registered model. The following sections of this page present a list of
registered names for each model category.

Language Modeling
-----------------

Components

.. autosummary::
    :nosignatures:

    AWDRNN
    BiLMEncoder
    LSTMPCellWithClip
    StandardRNN
    BigRNN

Pre-defined models

.. autosummary::
    :nosignatures:

    awd_lstm_lm_1150
    awd_lstm_lm_600
    standard_lstm_lm_200
    standard_lstm_lm_650
    standard_lstm_lm_1500
    big_rnn_lm_2048_512

Machine Translation
-------------------

.. autosummary::
    :nosignatures:

    Seq2SeqEncoder
    TransformerEncoder
    TransformerEncoderCell
    PositionwiseFFN

.. autosummary::
    :nosignatures:

    transformer_en_de_512

Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

Components

.. autosummary::
    :nosignatures:

    BERTModel
    BERTLayerNorm
    BERTEncoder
    BERTEncoderCell
    BERTPositionwiseFFN

Pre-defined models

.. autosummary::
    :nosignatures:

    bert_12_768_12
    bert_24_1024_16

Convolutional Encoder
---------------------

.. autosummary::
    :nosignatures:

    ConvolutionalEncoder

ELMo
----

Components

.. autosummary::
    :nosignatures:

    ELMoBiLM
    ELMoCharacterEncoder

Pre-defined models

.. autosummary::
    :nosignatures:

    elmo_2x1024_128_2048cnn_1xhighway
    elmo_2x2048_256_2048cnn_1xhighway
    elmo_2x4096_512_2048cnn_2xhighway

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
    GELU
    ISDense
    NCEDense
    SparseISDense
    SparseNCEDense

API Reference
-------------

.. automodule:: gluonnlp.model
    :members:
    :imported-members:
