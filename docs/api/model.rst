Gluon NLP Models
================

Gluon NLP Toolkit supplies models for common NLP tasks with pre-trained weights. By default,
all requested pre-trained weights are downloaded from public repo and stored in ~/.mxnet/models/.

Language Modeling
-----------------
.. automodule:: gluonnlp.model.language_model
    :members:

Building Blocks
===============

Gluon NLP Toolkit provides building blocks for NLP models.

Attention Cell
--------------
.. automodule:: gluonnlp.model.attention_cell

    .. autoclass:: gluonnlp.model.attention_cell.AttentionCell
        :members: __call__

    .. autoclass:: gluonnlp.model.attention_cell.MultiHeadAttentionCell
        :members: __call__

    .. autoclass:: gluonnlp.model.attention_cell.MLPAttentionCell
        :members: __call__

    .. autoclass:: gluonnlp.model.attention_cell.DotProductAttentionCell
        :members: __call__

Beam Search
-----------
.. automodule:: gluonnlp.model.beam_search

    .. autoclass:: gluonnlp.model.beam_search.BeamSearchScorer
        :members: __call__

    .. autoclass:: gluonnlp.model.beam_search.BeamSearchSampler
        :members: __call__

Other Modeling Utilities
------------------------
.. automodule:: gluonnlp.model.parameter
    :members:
.. automodule:: gluonnlp.model.utils
    :members:
.. automodule:: gluonnlp.model.block
    :members:
