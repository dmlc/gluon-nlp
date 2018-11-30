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
    BiLMEncoder
    LSTMPCellWithClip
    standard_lstm_lm_200
    standard_lstm_lm_650
    standard_lstm_lm_1500
    big_rnn_lm_2048_512
    StandardRNN
    get_model
    BigRNN

Machine Translation
-------------------

.. autosummary::
    :nosignatures:

    Seq2SeqEncoder
    TransformerEncoder
    TransformerEncoderCell
    PositionwiseFFN
    transformer_en_de_512

Bidirectional Encoder Representations from Transformers
-------------------------------------------------------

.. autosummary::
    :nosignatures:

    BERTModel
    BERTLayerNorm
    BERTEncoder
    BERTEncoderCell
    BERTPositionwiseFFN
    bert_12_768_12
    bert_24_1024_16

Convolutional Encoder
----------------------

.. autosummary::
    :nosignatures:

    ConvolutionalEncoder

ELMo
----------------------

.. autosummary::
    :nosignatures:

    ELMoBiLM
    ELMoCharacterEncoder

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
