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

"""Convolutional character encoder."""
__all__ = ['CharacterEncoder']

from mxnet import gluon, nd
from mxnet.gluon import nn

from .highway import Highway

class CharacterEncoder(gluon.Block):
    """
    We implement the convolutional character encoder proposed in the following work:

    @inproceedings{kim2016character,
    title={Character-Aware Neural Language Models.},
    author={Kim, Yoon and Jernite, Yacine and Sontag, David and Rush, Alexander M},
    booktitle={AAAI},
    pages={2741--2749},
    year={2016}
    }
    """
    def __init__(self,
                 embed_size,
                 num_filters,
                 ngram_filter_sizes,
                 conv_layer_activation=nn.Activation('relu'),
                 num_highway=None,
                 output_size=None,
                 **kwargs):
        super(CharacterEncoder, self).__init__(**kwargs)

        self._embed_size = embed_size
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation
        self._num_highway = num_highway
        self._output_size = output_size

        #
        # self._noutput = noutput
        # self._filters = filters
        # self._nembed = nembed
        # self._nhighway = nhighway
        # self._activation = activation
        # self._max_chars_per_token = max_chars_per_token
        # self._nfilters = len(filters)

        with self.name_scope():
            # self._embedding = nn.Embedding(input_dim=self._ninput, output_dim=self._nembed)
            self._convs = nn.HybridSequential()
            with self._convs.name_scope():
                for i, ngram_size in enumerate(self._ngram_filter_sizes):
                    # pylint: disable=unused-argument
                    self._convs.add(nn.Conv1D(in_channels=self._embed_size,
                                              channels=self._num_filters,
                                              kernel_size=ngram_size,
                                              use_bias=True))
            maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
            if self._num_highway:
                self._highways = Highway(maxpool_output_dim, self._num_highway, activation=self._activation)
            else:
                self._highways = None
            if self._output_size:
                self._projection = nn.Dense(in_units=maxpool_output_dim, units=self._output_size, use_bias=True)
            else:
                self._projection = None
                self._output_size = maxpool_output_dim

    def set_highway_bias(self):
        self._highways.set_bias()

    def forward(self, inputs, mask): # pylint: disable=arguments-differ
        """
        Forward computation for char_encoder
        """
        if mask is not None:
            inputs = inputs * mask.expand_dims(axis=-1)

        inputs = nd.transpose(inputs, axes=(0, 2, 1))

        filter_outputs = []
        for i, conv in enumerate(self._convs):
            filter_outputs.append(
                self._activation(conv(inputs).max(axis=2))[0]
            )

        print('filter_outputs:')
        print(filter_outputs)

        output = nd.concat(*filter_outputs, dim=0) if len(filter_outputs) > 1 else filter_outputs[0]

        if self._highways:
            output = self._highways(output)

        if self._projection:
            output = self._projection(output)

        return output


        # max_chars_per_token = self._max_chars_per_token
        #
        # character_embedding = self._embedding(inputs.reshape(-1, max_chars_per_token))
        #
        # activation = nn.Activation(self._activation)
        #
        # # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        # character_embedding = nd.swapaxes(character_embedding, 1, 2)
        # convs = []
        # for i in range(self._nfilters):
        #     conv = self._convs[i]
        #     convolved = conv(character_embedding)
        #     # (batch_size * sequence_length, n_filters for this width)
        #     # max pooling
        #     convolved = nd.max(convolved, axis=-1)
        #     convolved = activation(convolved)
        #     convs.append(convolved)
        #
        # # (batch_size * sequence_length, n_filters)
        # token_embedding = nd.concat(*convs, dim=-1)
        #
        # # apply the highway layers (batch_size * sequence_length, n_filters)
        # token_embedding = self._highways(token_embedding)
        #
        # # final projection  (batch_size * sequence_length, embedding_dim)
        # token_embedding = self._projection(token_embedding)
        #
        # # reshape to (sequence_length, batch_size, embedding_dim)
        # sequence_length, batch_size, _ = inputs.shape
        #
        # return token_embedding.reshape(sequence_length, batch_size, -1)
