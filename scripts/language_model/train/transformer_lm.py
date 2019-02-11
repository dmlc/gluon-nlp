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
__all__ = ['BERTRNN']

from mxnet.gluon import Block, nn
from mxnet import nd

from gluonnlp.model.utils import _get_rnn_layer


class BERTRNN(Block):
    """BERT based language model. Paper would be available soon.

    Parameters
    ----------
    embedding : BERTMaskedModel
        The BERT structure to use.
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
    hidden_size_last : int
        Number of last hidden units for RNN.
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
    drop_l : float
        Dropout rate to use on the latent layer.
    num_experts : int
        Number of experts in mixture of softmax.
    upperbound_fixed_layer : int
        Number of layers in BERT with the parameters fixed.

    Inputs:
        - **inputs**: input sequence tensor, shape (batch_size, seq_length)
        - **begin_state**: initial recurrent state tensor with length equals to num_layers.
            the initial state with shape `(1, batch_size, num_hidden)`
        - **token_types**: input token type tensor, shape (batch_size, seq_length).
            If the inputs contain two sequences, then the token type of the first
            sequence differs from that of the second one.
        - **valid_length**: optional tensor of input sequence valid lengths, shape (batch_size,)
        - **masked_positions**: optional tensor of position of tokens for masked LM decoding,
            shape (batch_size, num_masked_positions).

    Outputs:
        - **out**: output tensor with shape `(sequence_length, batch_size, input_size)`
            when `layout` is "TNC".
        - **out_states**: output recurrent state tensor with length equals to num_layers.
            the state with shape `(1, batch_size, num_hidden)`
        - **encoded_raw**: The list of outputs of the model's encoder with length equals to
            num_layers. the shape of every encoder's output
            `(sequence_length, batch_size, num_hidden)`
        - **encoded_dropped**: The list of outputs with dropout of the model's encoder with
            length equals to num_layers. The shape of every encoder's dropped output
            `(sequence_length, batch_size, num_hidden)`
    """

    def __init__(self, embedding, mode, vocab_size, embed_size=300, hidden_size=1150, hidden_size_last=650,
                 num_layers=3, tie_weights=True, dropout=0.4, weight_drop=0.5, drop_h=0.2,
                 drop_i=0.55, drop_e=0.1, drop_l=0.29, num_experts=15,
                 upperbound_fixed_layer=22, **kwargs):
        super(BERTRNN, self).__init__(**kwargs)
        self._mode = mode
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._hidden_size_last = hidden_size_last
        self._num_layers = num_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
        self._drop_e = drop_e
        self._drop_l = drop_l
        self._num_experts = num_experts
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights

        #bert specific setting
        self._embed_size = embedding.encoder._units

        with self.name_scope():
            self.embedding = embedding
            for cell in self.embedding.encoder.transformer_cells[:upperbound_fixed_layer]:
                cell.collect_params().setattr('grad_req', 'null')
            self.encoder = self._get_encoder()
            self.prior = self._get_prior()
            self.latent = self._get_latent()
            self.decoder = self._get_decoder()

    def _get_encoder(self):
        encoder = nn.Sequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(_get_rnn_layer(self._mode, 1, self._embed_size if l == 0 else
                self._hidden_size, self._hidden_size if
                                           l != self._num_layers - 1 else self._hidden_size_last,
                                           0, self._weight_drop))
        return encoder

    def _get_prior(self):
        prior = nn.HybridSequential()
        with prior.name_scope():
            prior.add(nn.Dense(self._num_experts, in_units=self._hidden_size_last,
                               use_bias=False, flatten=False))
        return prior

    def _get_latent(self):
        latent = nn.HybridSequential()
        with latent.name_scope():
            latent.add(nn.Dense(self._num_experts * self._embed_size, 'tanh',
                                in_units=self._hidden_size_last, flatten=False))
        return latent

    def _get_decoder(self):
        decoder = nn.HybridSequential()
        with decoder.name_scope():
            if self._tie_weights:
                #BERT specific setting
                decoder.add(nn.Dense(self._vocab_size, flatten=False,
                                     params=self.embedding.word_embed.params))
            else:
                decoder.add(nn.Dense(self._vocab_size, flatten=False))
        return decoder

    def begin_state(self, *args, **kwargs):
        return [c.begin_state(*args, **kwargs) for c in self.encoder]

    def __call__(self, inputs, begin_state=None, token_types=None, valid_length=None, masked_positions=None, *args, **kwargs):
        return super(BERTRNN, self).__call__(inputs, begin_state, token_types, valid_length, masked_positions)

    def forward(self, inputs, begin_state=None, token_types=None, valid_length=None, masked_positions=None):  # pylint: disable=arguments-differ
        """Implement the forward computation that the awd language model and cache model use.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(sequence_length, batch_size)`
            when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers.
            the initial state with shape `(1, batch_size, num_hidden)`
        token_types: NDArray
            input token type tensor, shape (batch_size, seq_length).
            If the inputs contain two sequences, then the token type of the first
            sequence differs from that of the second one.
        valid_length: NDArray
            optional tensor of input sequence valid lengths, shape (batch_size,)
        masked_positions: optional tensor of position of tokens for masked LM decoding,
            shape (batch_size, num_masked_positions).

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
        batch_size = inputs.shape[1]
        inputs = nd.transpose(inputs, axes=(1,0))
        if token_types is None:
            token_types = nd.zeros_like(inputs)
        encoded = self.embedding(inputs, token_types=token_types,
                                 valid_length=valid_length, masked_positions=masked_positions)
        encoded = nd.transpose(encoded, axes=(1,0,2))
        encoded = nd.Dropout(encoded, p=self._drop_i, axes=(0,))
        if not begin_state:
            begin_state = self.begin_state(batch_size=batch_size)
        out_states = []
        encoded_raw = []
        encoded_dropped = []
        for i, (e, s) in enumerate(zip(self.encoder, begin_state)):
            encoded, state = e(encoded, s)
            encoded_raw.append(encoded)
            out_states.append(state)
            if i != len(self.encoder) - 1:
                encoded = nd.Dropout(encoded, p=self._drop_h, axes=(0,))
                encoded_dropped.append(encoded)
        encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        encoded_dropped.append(encoded)
        #use mos
        latent = nd.Dropout(self.latent(encoded), p=self._drop_l, axes=(0,))
        logit = self.decoder(latent.reshape(-1, self._embed_size))

        prior_logit = self.prior(encoded).reshape(-1, self._num_experts)
        prior = nd.softmax(prior_logit, axis=-1)

        prob = nd.softmax(logit.reshape(-1, self._vocab_size), axis=-1)
        prob = prob.reshape(-1, self._num_experts, self._vocab_size)
        prob = (prob * prior.expand_dims(2).broadcast_to(prob.shape)).sum(axis=1)

        out = nd.log(nd.add(prob, 1e-8)).reshape(-1, batch_size, self._vocab_size)

        return out, out_states, encoded_raw, encoded_dropped
