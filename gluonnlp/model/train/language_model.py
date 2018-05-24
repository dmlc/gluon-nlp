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
"""Language models."""
__all__ = ['AWDRNN', 'StandardRNN']

import os
import warnings

from mxnet.gluon.model_zoo.model_store import get_model_file
from mxnet import init, nd, cpu, autograd
from mxnet.gluon import nn, Block
from mxnet.gluon.model_zoo import model_store

from gluonnlp.model.infer import AWDRNN, StandardRNN
from gluonnlp.model.utils import _get_rnn_layer
from gluonnlp.model.utils import apply_weight_drop
from gluonnlp.data.utils import _load_pretrained_vocab


class AWDRNN(AWDRNN):
    """AWD language model by salesforce.

    Reference: https://github.com/salesforce/awd-lstm-lm

    License: BSD 3-Clause

    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
    num_layers : int
        Number of RNN layers.
    tie_weights : bool, default False
        Whether to tie the weight matrices of output dense layer and input embedding layer.
    dropout : float
        Dropout rate to use for encoder output.
    weight_drop : float
        Dropout rate to use on encoder h2h weights.
    drop_h : float
        Dropout rate to on the output of intermediate layers of encoder.
    drop_i : float
        Dropout rate to on the output of embedding.
    drop_e : float
        Dropout rate to use on the embedding layer.
    """
    def __init__(self, mode, vocab_size, embed_size, hidden_size, num_layers,
                 tie_weights, dropout, weight_drop, drop_h,
                 drop_i, drop_e, **kwargs):
        super(AWDRNN, self).__init__(mode, vocab_size, embed_size, hidden_size, num_layers,
                 tie_weights, dropout, weight_drop, drop_h,
                 drop_i, drop_e, **kwargs)

    def forward(self, inputs, begin_state=None):
        """Implement the forward computation that the awd language model and cache model use.

        Parameters
        ----------
        inputs : NDArray
            The training dataset.
        begin_state : list
            The initial hidden states.

        Returns
        -------
        out: NDArray
            The output of the model.
        out_states: list
            The list of output states of the model's encoder.
        encoded_raw: list
            The list of outputs of the model's encoder.
        encoded_dropped: list
            The list of outputs with dropout of the model's encoder.
        """
        encoded = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state(batch_size=inputs.shape[1])
        out_states = []
        encoded_raw = []
        encoded_dropped = []
        for i, (e, s) in enumerate(zip(self.encoder, begin_state)):
            encoded, state = e(encoded, s)
            encoded_raw.append(encoded)
            out_states.append(state)
            if self._drop_h and i != len(self.encoder) - 1:
                encoded = nd.Dropout(encoded, p=self._drop_h, axes=(0,))
                encoded_dropped.append(encoded)
        if self._dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        encoded_dropped.append(encoded)
        with autograd.predict_mode():
            out = self.decoder(encoded)
        return out, out_states, encoded_raw, encoded_dropped

class StandardRNN(StandardRNN):
    """Standard RNN language model.

    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
    num_layers : int
        Number of RNN layers.
    dropout : float
        Dropout rate to use for encoder output.
    tie_weights : bool, default False
        Whether to tie the weight matrices of output dense layer and input embedding layer.
    """
    def __init__(self, **kwargs):
        if super(StandardRNN, self).tie_weights:
            assert super(StandardRNN, self).embed_size == super(StandardRNN, self).hidden_size, 'Embedding dimension must be equal to ' \
                                              'hidden dimension in order to tie weights. ' \
                                              'Got: emb: {}, hid: {}.'.format(super(StandardRNN, self).embed_size,
                                                                              super(StandardRNN, self).hidden_size)
        super(StandardRNN, self).__init__(**kwargs)

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        encoded = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state(batch_size=inputs.shape[1])
        encoded_raw = []
        encoded_dropped = []
        encoded, state = self.encoder(encoded, begin_state)
        encoded_raw.append(encoded)
        if self._dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        out = self.decoder(encoded)
        return out, state, encoded_raw, encoded_dropped
