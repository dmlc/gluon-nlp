# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=wildcard-import, arguments-differ
r"""Module for pre-defined NLP models.

This module contains definitions for the following model architectures:
-  `AWD`_

You can construct a model with random weights by calling its constructor. Because NLP models
are tied to vocabularies, you can either specify a dataset name to load and use the vocabulary
of that dataset:

.. code-block:: python

    import gluonnlp as nlp
    awd, vocab = nlp.model.awd_lstm_lm_1150(dataset_name='wikitext-2')

or directly specify a vocabulary object:

.. code-block:: python

    awd, vocab = nlp.model.awd_lstm_lm_1150(None, vocab=custom_vocab)

We provide pre-trained models for all the listed models.
These models can constructed by passing ``pretrained=True``:

.. code-block:: python

    awd, vocab = nlp.model.awd_lstm_lm_1150(dataset_name='wikitext-2'
                                            pretrained=True)

.. _AWD: https://arxiv.org/abs/1404.5997
"""

from .language_model import *

from .beam_search import *

from .attention_cell import *

from .utils import *

from .parameter import *

from .block import *

from .highway import *

from .convolutional_encoder import *

from . import train

__all__ = language_model.__all__ + beam_search.__all__ + attention_cell.__all__ + \
          utils.__all__ + parameter.__all__ + block.__all__ + highway.__all__ + \
          convolutional_encoder.__all__ + ['get_model'] + ['train']


def get_model(name, dataset_name='wikitext-2', **kwargs):
    """Returns a pre-defined model by name.

    Parameters
    ----------
    name : str
        Name of the model.
    dataset_name : str or None, default 'wikitext-2'.
        The dataset name on which the pretrained model is trained.
        Options are 'wikitext-2'. If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    Block
        The model.
    """
    models = {'standard_lstm_lm_200': standard_lstm_lm_200,
              'standard_lstm_lm_650': standard_lstm_lm_650,
              'standard_lstm_lm_1500': standard_lstm_lm_1500,
              'awd_lstm_lm_1150': awd_lstm_lm_1150,
              'awd_lstm_lm_600': awd_lstm_lm_600}
    name = name.lower()
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s'%(
                name, '\n\t'.join(sorted(models.keys()))))
    kwargs['dataset_name'] = dataset_name
    return models[name](**kwargs)
