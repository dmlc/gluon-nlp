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


-  `ELMo`_

You can construct a predefined ELMo model structure:

.. code-block:: python

    import gluonnlp as nlp
    elmo = nlp.model.elmo_2x1024_128_2048cnn_1xhighway(dataset_name='gbw')

You can also get a ELMo model with pretrained parameters:

.. code-block:: python

    import gluonnlp as nlp
    elmo = nlp.model.elmo_2x1024_128_2048cnn_1xhighway(dataset_name='gbw', pretrained=True)

.. _ELMo: https://arxiv.org/pdf/1802.05365.pdf
"""


from . import (attention_cell, sequence_sampler, block, convolutional_encoder,
               highway, language_model, parameter, sampled_block, train, utils, bilm_encoder,
               lstmpcellwithclip, elmo)
from .attention_cell import *
from .sequence_sampler import *
from .block import *
from .convolutional_encoder import *
from .seq2seq_encoder_decoder import *
from .translation import *
from .transformer import *
from .bert import *
from .highway import *
from .language_model import *
from .parameter import *
from .sampled_block import *
from .utils import *
from .bilm_encoder import BiLMEncoder
from .lstmpcellwithclip import LSTMPCellWithClip
from .elmo import *

__all__ = language_model.__all__ + sequence_sampler.__all__ + attention_cell.__all__ + \
          utils.__all__ + parameter.__all__ + block.__all__ + highway.__all__ + \
          convolutional_encoder.__all__ + sampled_block.__all__ + ['get_model'] + ['train'] + \
          bilm_encoder.__all__ + lstmpcellwithclip.__all__ + elmo.__all__ + \
          seq2seq_encoder_decoder.__all__ + transformer.__all__ + bert.__all__


def get_model(name, dataset_name='wikitext-2', **kwargs):
    """Returns a pre-defined model by name.

    Parameters
    ----------
    name : str
        Name of the model.
    dataset_name : str or None, default 'wikitext-2'.
        The dataset name on which the pre-trained model is trained.
        For language model, options are 'wikitext-2'.
        For ELMo, Options are 'gbw' and '5bw'.
        'gbw' represents 1 Billion Word Language Model Benchmark
        http://www.statmt.org/lm-benchmark/;
        '5bw' represents a dataset of 5.5B tokens consisting of
        Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008-2012 (3.6B).
        If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
        None Vocabulary object is required with the ELMo model.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab, (optional) gluonnlp.Vocab
    """
    models = {'standard_lstm_lm_200' : standard_lstm_lm_200,
              'standard_lstm_lm_650' : standard_lstm_lm_650,
              'standard_lstm_lm_1500': standard_lstm_lm_1500,
              'awd_lstm_lm_1150': awd_lstm_lm_1150,
              'awd_lstm_lm_600': awd_lstm_lm_600,
              'big_rnn_lm_2048_512': big_rnn_lm_2048_512,
              'elmo_2x1024_128_2048cnn_1xhighway': elmo_2x1024_128_2048cnn_1xhighway,
              'elmo_2x2048_256_2048cnn_1xhighway': elmo_2x2048_256_2048cnn_1xhighway,
              'elmo_2x4096_512_2048cnn_2xhighway': elmo_2x4096_512_2048cnn_2xhighway,
              'transformer_en_de_512': transformer_en_de_512,
              'bert_12_768_12'       : bert_12_768_12,
              'bert_24_1024_16'      : bert_24_1024_16}
    name = name.lower()
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s'%(
                name, '\n\t'.join(sorted(models.keys()))))
    kwargs['dataset_name'] = dataset_name
    return models[name](**kwargs)
