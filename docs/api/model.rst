gluonnlp.model
==============

Gluon NLP Toolkit supplies models for common NLP tasks with pre-trained weights. By default,
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
    StandardRNN

Attention Cell
--------------

.. autosummary::
    :nosignatures:

    AttentionCell
    MultiHeadAttentionCell
    MLPAttentionCell
    DotProductAttentionCell

Beam Search
-----------

.. autosummary::
    :nosignatures:

    BeamSearchScorer
    BeamSearchSampler

Other Modeling Utilities
------------------------

.. autosummary::
    :nosignatures:

    WeightDropParameter
    apply_weight_drop
    L2Normalization

API Reference
-------------

.. autofunction:: awd_lstm_lm_1150
.. autofunction:: awd_lstm_lm_600
.. autoclass:: AWDRNN
.. autofunction:: standard_lstm_lm_200
.. autofunction:: standard_lstm_lm_650
.. autofunction:: standard_lstm_lm_1500
.. autoclass:: StandardRNN

.. autoclass:: gluonnlp.model.AttentionCell
    :members: __call__

.. autoclass:: gluonnlp.model.MultiHeadAttentionCell
    :members: __call__

.. autoclass:: gluonnlp.model.MLPAttentionCell
    :members: __call__

.. autoclass:: gluonnlp.model.DotProductAttentionCell
    :members: __call__

.. autoclass:: gluonnlp.model.BeamSearchScorer
    :members: __call__

.. autoclass:: gluonnlp.model.BeamSearchSampler
    :members: __call__

.. autoclass:: gluonnlp.model.WeightDropParameter

.. autoclass:: gluonnlp.model.L2Normalization

.. autofunction:: gluonnlp.model.apply_weight_drop
