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
__all__ = ['RNNCellLayer', 'L2Normalization', 'CharacterLevelCNNEmbedding', 'PredefinedEmbedding',
           'BiDAFEmbedding']

from mxnet import ndarray
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
    """Normalize the input array by dividing the L2 norm along the given axis.

    ..code

        out = data / (sqrt(sum(data**2, axis)) + eps)

    Parameters
    ----------
    axis : int, default -1
        The axis to compute the norm value.
    eps : float, default 1E-6
        The epsilon value to avoid dividing zero
    """
    def __init__(self, axis=-1, eps=1E-6, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self._axis = axis
        self._eps = eps

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        ret = F.broadcast_div(x, F.norm(x, axis=self._axis, keepdims=True) + self._eps)
        return ret


class CharacterLevelCNNEmbedding(HybridBlock):
    """
    CharacterLevelCNNEmbedding takes data and embed it using 2D convolution layers. Quantity of
    Conv2D block is equal to len of channels and kernel_size lists, which should be same.

    Each item in the data is an index of a character, drawn from a vocab of `vocab_size`.
    Input layout: NTC, where
        N - batch_size = # of examples,
        T - seq_len    = # of words in each example,
        C - channel    = # char in each word

    Output layout: batch_size x seq_len x channels, where
        batch_size = # of examples,
        seq_len    = # words in each example,
        channels   = # channels equal to `channels` parameter passed to `__init__()`
    """
    def __init__(self, channels, kernel_sizes, padding, vocab_size, char_embedding_size=8,
                 keep_prob=0.2, prefix=None, params=None):
        super(CharacterLevelCNNEmbedding, self).__init__()

        assert len(channels) == len(kernel_sizes)
        self.net = nn.HybridSequential(prefix=prefix, params=params)

        with self.net.name_scope():
            self.net.add(nn.Embedding(input_dim=vocab_size, output_dim=char_embedding_size))

            for channel, kernel in zip(channels, kernel_sizes):
                self.net.add(nn.Dropout(rate=keep_prob))
                self.net.add(nn.Conv2D(layout='NHWC', in_channels=char_embedding_size,
                                       strides=[1, 1], channels=channel, kernel_size=[1, kernel],
                                       padding=padding))

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        network_output = self.net.hybrid_forward(F, x)
        relu_output = F.relu(network_output)
        return F.max(relu_output, axis=2)


class PredefinedEmbedding(HybridBlock):
    """
    PredefinedEmbedding embeds data using pre-trained embedding like glove or others. Each item in
    the data should be an index from a vocab, that uses the same embedding passed to `__init__()`.

    Input layout:  batch_size x seq_length, where
        batch_size = # of examples,
        seq_len    = # words in each example,

    Output layout: batch_size x seq_len x embed_size, where
        batch_size = # of examples,
        seq_len    = # words in each example,
        embed_size = dimensionality of the embedding, passed to `__init__()`
    """
    def __init__(self, embedding, prefix=None, params=None):
        super(PredefinedEmbedding, self).__init__(prefix=prefix, params=params)
        input_dim, output_dim = embedding.idx_to_vec.shape
        self.embedding_data = embedding.idx_to_vec
        self.embedding = nn.Embedding(input_dim, output_dim)

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        return self.embedding.hybrid_forward(F, x, weight=self.embedding_data)


class BiDAFEmbedding(HybridBlock):
    """
    BiDAFEmbedding uses CharacterLevelCNNEmbedding and PredefinedEmbedding to create a word + char
    level embeddings. It concats the resulting embedding into a single dataset by embedding
    dimension. The embedding sizes should be same for both datasets.

    Input: 2 datasets: first for CharacterLevelCNNEmbedding and second for PredefinedEmbedding
    Shapes of both datasets should match to these embeddings

    Output layout: batch_size x seq_len x embed_size, where
        batch_size = # of examples,
        seq_len    = # words in each example,
        embed_size = dimensionality of the embedding, equal to 2 * embedding_size of source data
    """
    def __init__(self, word_embedding_source, channels, kernel_sizes, padding, char_vocab_size,
                 char_embedding_size=8, keep_prob=0.2, prefix=None, params=None):
        super(BiDAFEmbedding, self).__init__(prefix=prefix, params=params)

        self.cnn_embedding = CharacterLevelCNNEmbedding(channels, kernel_sizes,
                                                        padding, char_vocab_size,
                                                        char_embedding_size, keep_prob,
                                                        prefix, params)
        self.word_embedding = PredefinedEmbedding(word_embedding_source, prefix, params)

    def hybrid_forward(self, F, char_level, word_level):  # pylint: disable=arguments-differ
        char_level_embedding = self.cnn_embedding.hybrid_forward(F, char_level)
        word_level_embedding = self.word_embedding.hybrid_forward(F, word_level)

        return F.concat(char_level_embedding, word_level_embedding, dim=2)