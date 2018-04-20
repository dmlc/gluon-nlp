Gluon NLP Models
================

Gluon NLP Toolkit supplies models for common NLP tasks with pre-trained weights. By default,
all requested pre-trained weights are downloaded from public repo and stored in ~/.mxnet/models/.

Language Modeling
-----------------
.. automodule:: gluonnlp.model.language_model
    :members:

Modeling Utilities
------------------
.. automodule:: gluonnlp.model.parameter
    :members:
.. automodule:: gluonnlp.model.utils
    :members:
.. automodule:: gluonnlp.model.block
    :members:
.. automodule:: gluonnlp.model.attention_cell
    :members:
.. automodule:: gluonnlp.model.beam_search

    .. autoclass:: gluonnlp.model.beam_search.BeamSearchScorer
        :members: __call__

    .. autoclass:: gluonnlp.model.beam_search.BeamSearchSampler
        :members: __call__
