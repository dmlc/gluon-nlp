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

"""Building blocks and utility for models."""
__all__ = ['RNNCellLayer', 'L2Normalization']

from mxnet import ndarray, initializer
from mxnet.gluon import Block, HybridBlock, nn


class RNNCellLayer(Block):
    """A block that takes an rnn cell and makes it act like rnn layer.

    Parameters
    ----------
    rnn_cell : Cell
        The cell to wrap into a layer-like block.
    layout : str, default 'TNC'
        The output layout of the layer.
    """
    def __init__(self, rnn_cell, layout='TNC', **kwargs):
        super(RNNCellLayer, self).__init__(**kwargs)
        self.cell = rnn_cell
        assert layout == 'TNC' or layout == 'NTC', \
            'Invalid layout %s; must be one of ["TNC" or "NTC"]'%layout
        self._layout = layout
        self._axis = layout.find('T')
        self._batch_axis = layout.find('N')

    def forward(self, inputs, states=None): # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        batch_size = inputs.shape[self._batch_axis]
        skip_states = states is None
        if skip_states:
            states = self.cell.begin_state(batch_size, ctx=inputs.context)
        if isinstance(states, ndarray.NDArray):
            states = [states]
        for state, info in zip(states, self.cell.state_info(batch_size)):
            if state.shape != info['shape']:
                raise ValueError(
                    'Invalid recurrent state shape. Expecting %s, got %s.'%(
                        str(info['shape']), str(state.shape)))
        states = sum(zip(*((j for j in i) for i in states)), ())
        outputs, states = self.cell.unroll(
            inputs.shape[self._axis], inputs, states,
            layout=self._layout, merge_outputs=True)

        if skip_states:
            return outputs
        return outputs, states


class L2Normalization(HybridBlock):
    def __init__(self, axis=-1, eps=1E-6, prefix=None, params=None):
        super(L2Normalization, self).__init__(prefix=prefix, params=params)
        self._axis = axis
        self._eps = eps

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        ret = F.broadcast_div(x, F.norm(x, axis=self._axis, keepdims=True) + self._eps)
        return ret


class CharacterLevelCNNEmbedding(HybridBlock):
    """
    A block that takes text and embed it using convolution neural network
    """
    def __init__(self, channels, kernel_sizes, padding, prefix=None, params=None):
        super(CharacterLevelCNNEmbedding, self).__init__()

        self.net = nn.HybridSequential(prefix=prefix, params=params)
        with self.net.name_scope():
            for channel, kernel in zip(channels, kernel_sizes):
                self.net.add(nn.Conv1D(channels=channel, kernel_size=kernel, padding=padding))

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        return self.net.hybrid_forward(F, x)


class PredefinedEmbedding(HybridBlock):
    """
    A block that takes text and embeds it using Glove embedding.
    """
    def __init__(self, embedding, prefix=None, params=None):
        super(PredefinedEmbedding, self).__init__(prefix=prefix, params=params)
        input_dim, output_dim = embedding.idx_to_vec.shape
        self.embedding_data = embedding.idx_to_vec
        self.embedding = nn.Embedding(input_dim, output_dim)

    #def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False, force_reinit=False):
    #    super(PredefinedEmbedding, self).initialize()
    #    self.embedding.initialize()
    #    self.embedding.weight.set_data(self.embedding_data)

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        return F.transpose(self.embedding.hybrid_forward(F, x, weight=self.embedding_data))


class BiFAREmbedding(HybridBlock):
    """
    An embedding for BiFAR model
    """
    def __init__(self, channels, kernel_sizes, padding, char_embedding_source, prefix=None,
                 params=None):
        super(BiFAREmbedding, self).__init__(prefix=prefix, params=params)
        self.cnn_embedding = CharacterLevelCNNEmbedding(channels, kernel_sizes, padding)
        self.word_embedding = PredefinedEmbedding(char_embedding_source)

    def hybrid_forward(self, F, x):
        pass