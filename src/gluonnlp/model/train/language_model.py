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
"""Language models for training."""
__all__ = ['AWDRNN', 'StandardRNN', 'BigRNN']

from mxnet import init, nd, autograd
from mxnet.gluon import nn, Block, contrib, rnn

from ..utils import _get_rnn_layer, apply_weight_drop
from ..sampled_block import ISDense, SparseISDense

class AWDRNN(Block):
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
    def __init__(self, mode, vocab_size, embed_size=400, hidden_size=1150, num_layers=3,
                 tie_weights=True, dropout=0.4, weight_drop=0.5, drop_h=0.2,
                 drop_i=0.65, drop_e=0.1, **kwargs):
        super(AWDRNN, self).__init__(**kwargs)
        self._mode = mode
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
        self._drop_e = drop_e
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding_block = nn.Embedding(self._vocab_size, self._embed_size,
                                           weight_initializer=init.Uniform(0.1))
            if self._drop_e:
                apply_weight_drop(embedding_block, 'weight', self._drop_e, axes=(1,))
            embedding.add(embedding_block)
            if self._drop_i:
                embedding.add(nn.Dropout(self._drop_i, axes=(0,)))
        return embedding

    def _get_encoder(self):
        encoder = nn.Sequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(_get_rnn_layer(self._mode, 1, self._embed_size if l == 0 else
                                           self._hidden_size, self._hidden_size if
                                           l != self._num_layers - 1 or not self._tie_weights
                                           else self._embed_size, 0, self._weight_drop))
        return encoder

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            if self._tie_weights:
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return [c.begin_state(*args, **kwargs) for c in self.encoder]

    def state_info(self, *args, **kwargs):
        return [c.state_info(*args, **kwargs) for c in self.encoder]

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Implement the forward computation that the awd language model and cache model use.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(sequence_length, batch_size)`
            when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers.
            the initial state with shape `(1, batch_size, num_hidden)`

        Returns
        --------
        out: NDArray
            output tensor with shape `(sequence_length, batch_size, input_size)`
            when `layout` is "TNC".
        out_states: list
            output recurrent state tensor with length equals to num_layers.
            the state with shape `(1, batch_size, num_hidden)`
        encoded_raw: list
            The list of outputs of the model's encoder with length equals to num_layers.
            the shape of every encoder's output `(sequence_length, batch_size, num_hidden)`
        encoded_dropped: list
            The list of outputs with dropout of the model's encoder with length equals
            to num_layers. The shape of every encoder's dropped output
            `(sequence_length, batch_size, num_hidden)`
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


class StandardRNN(Block):
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
    def __init__(self, mode, vocab_size, embed_size, hidden_size,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        if tie_weights:
            assert embed_size == hidden_size, 'Embedding dimension must be equal to ' \
                                              'hidden dimension in order to tie weights. ' \
                                              'Got: emb: {}, hid: {}.'.format(embed_size,
                                                                              hidden_size)
        super(StandardRNN, self).__init__(**kwargs)
        self._mode = mode
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._tie_weights = tie_weights
        self._vocab_size = vocab_size

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(nn.Embedding(self._vocab_size, self._embed_size,
                                       weight_initializer=init.Uniform(0.1)))
            if self._dropout:
                embedding.add(nn.Dropout(self._dropout))
        return embedding

    def _get_encoder(self):
        return _get_rnn_layer(self._mode, self._num_layers, self._embed_size,
                              self._hidden_size, self._dropout, 0)

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            if self._tie_weights:
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def state_info(self, *args, **kwargs):
        return self.encoder.state_info(*args, **kwargs)

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(sequence_length, batch_size)`
            when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers-1.
            the initial state with shape `(num_layers, batch_size, num_hidden)`

        Returns
        --------
        out: NDArray
            output tensor with shape `(sequence_length, batch_size, input_size)`
            when `layout` is "TNC".
        out_states: list
            output recurrent state tensor with length equals to num_layers-1.
            the state with shape `(num_layers, batch_size, num_hidden)`
        encoded_raw: list
            The list of last output of the model's encoder.
            the shape of last encoder's output `(sequence_length, batch_size, num_hidden)`
        encoded_dropped: list
            The list of last output with dropout of the model's encoder.
            the shape of last encoder's dropped output `(sequence_length, batch_size, num_hidden)`
        """
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

class BigRNN(Block):
    """Big language model with LSTMP and importance sampling.

    Reference: https://github.com/rafaljozefowicz/lm

    License: MIT

    Parameters
    ----------
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for LSTMP.
    num_layers : int
        Number of LSTMP layers.
    projection_size : int
        Number of projection units for LSTMP.
    num_sampled : int
        Number of sampled classes for the decoder.
    embed_dropout : float
        Dropout rate to use for embedding output.
    encoder_dropout : float
        Dropout rate to use for encoder output.
    sparse_weight : bool
        Whether to use RewSparseNDArray for weights of input and output embeddings.
    sparse_grad : bool
        Whether to use RowSparseNDArray for the gradients w.r.t.
        weights of input and output embeddings.

    .. note: If `sparse_grad` is set to True, the gradient w.r.t input and output
             embeddings will be sparse. Only a subset of optimizers support
             sparse gradients, including SGD, AdaGrad and Adam.
             By default `lazy_update` is turned on for these optimizers,
             which may perform differently from standard updates.
             For more details, please check the Optimization API at:
             https://mxnet.incubator.apache.org/api/python/optimization/optimization.html

    .. note: If `sparse_weight` is set to True, the parameters in the embedding block and
             decoder block will be stored in row_sparse format, which helps reduce memory
             consumption and communication overhead during multi-GPU training. However,
             sparse parameters cannot be shared with other blocks, nor could we hybridize
             a block containinng sparse parameters.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 projection_size, num_sampled, embed_dropout=0.0, encode_dropout=0.0,
                 sparse_weight=True, sparse_grad=True, **kwargs):
        super(BigRNN, self).__init__(**kwargs)
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._projection_size = projection_size
        self._num_layers = num_layers
        self._embed_dropout = embed_dropout
        self._encode_dropout = encode_dropout
        self._vocab_size = vocab_size
        self._num_sampled = num_sampled
        self._sparse_weight = sparse_weight
        self._sparse_grad = sparse_grad
        if self._sparse_weight:
            assert self._sparse_grad, 'Dense grad with sparse weight is not supported.'

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        prefix = 'embedding0_'
        if self._sparse_weight:
            embedding = nn.Sequential(prefix=prefix)
        else:
            embedding = nn.HybridSequential(prefix=prefix)
        with embedding.name_scope():
            if self._sparse_weight:
                # sparse embedding has both sparse weight and sparse grad
                embed = contrib.nn.SparseEmbedding(self._vocab_size, self._embed_size,
                                                   prefix=prefix)
            else:
                embed = nn.Embedding(self._vocab_size, self._embed_size, prefix=prefix,
                                     sparse_grad=self._sparse_grad)
            embedding.add(embed)
            if self._embed_dropout:
                embedding.add(nn.Dropout(self._embed_dropout))
        return embedding

    def _get_encoder(self):
        block = rnn.HybridSequentialRNNCell()
        with block.name_scope():
            for _ in range(self._num_layers):
                block.add(contrib.rnn.LSTMPCell(self._hidden_size, self._projection_size))
                if self._encode_dropout:
                    block.add(rnn.DropoutCell(self._encode_dropout))
        return block

    def _get_decoder(self):
        prefix = 'decoder0_'
        if self._sparse_weight:
            # sparse IS Dense has both sparse weight and sparse grad
            block = SparseISDense(self._vocab_size, self._num_sampled,
                                  self._projection_size, remove_accidental_hits=True,
                                  prefix=prefix)
        else:
            block = ISDense(self._vocab_size, self._num_sampled,
                            self._projection_size, remove_accidental_hits=True,
                            prefix=prefix, sparse_grad=self._sparse_grad)
        return block

    def begin_state(self, **kwargs):
        return self.encoder.begin_state(**kwargs)

    def forward(self, inputs, label, begin_state, sampled_values): # pylint: disable=arguments-differ
        """Defines the forward computation.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(sequence_length, batch_size)`
            when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers*2.
            For each layer the two initial states have shape `(batch_size, num_hidden)`
            and `(batch_size, num_projection)`
        sampled_values : list
            a list of three tensors for `sampled_classes` with shape `(num_samples,)`,
            `expected_count_sampled` with shape `(num_samples,)`, and
            `expected_count_true` with shape `(sequence_length, batch_size)`.

        Returns
        --------
        out : NDArray
            output tensor with shape `(sequence_length, batch_size, 1+num_samples)`
            when `layout` is "TNC".
        out_states : list
            output recurrent state tensor with length equals to num_layers*2.
            For each layer the two initial states have shape `(batch_size, num_hidden)`
            and `(batch_size, num_projection)`
        new_target : NDArray
            output tensor with shape `(sequence_length, batch_size)`
            when `layout` is "TNC".
        """
        encoded = self.embedding(inputs)
        length = inputs.shape[0]
        batch_size = inputs.shape[1]
        encoded, out_states = self.encoder.unroll(length, encoded, begin_state,
                                                  layout='TNC', merge_outputs=True)
        out, new_target = self.decoder(encoded, sampled_values, label)
        out = out.reshape((length, batch_size, -1))
        new_target = new_target.reshape((length, batch_size))
        return out, out_states, new_target
