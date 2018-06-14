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

from __future__ import absolute_import
from __future__ import print_function

__all__ = ['CharacterEncoder']

from mxnet import gluon, nd
from mxnet.gluon import nn

from .highway import Highway


class CharacterEncoder(gluon.Block):
    r"""

    We implement the convolutional character encoder proposed in the following work::

        @inproceedings{kim2016character,
         title={Character-Aware Neural Language Models.},
         author={Kim, Yoon and Jernite, Yacine and Sontag, David and Rush, Alexander M},
         booktitle={AAAI},
         pages={2741--2749},
         year={2016}
        }

    Parameters
    ----------
    embed_size : int
        The input dimension to the encoder.
    num_filters: int
        The output dimension for each convolutional layer,
        which is the number of the filters learned by the layer.
    ngram_filter_sizes: Tuple[int]
        The size of each convolutional layer,
        and len(ngram_filter_sizes) equals to the number of convolutional layers.
    conv_layer_activation: nn.Activation
        Activation function to be used after convolutional layer.
    num_highway: int
        The number of layers of the Highway layer.
    output_size: int
        The output dimension after conducting the convolutions and max pooling,
        and apply highways, as well as linear projection.

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
        self._num_highway = num_highway
        self._output_size = output_size

        with self.name_scope():
            self._convs = nn.HybridSequential()
            with self._convs.name_scope():
                for _, ngram_size in enumerate(self._ngram_filter_sizes):
                    self._convs.add(nn.Conv1D(in_channels=self._embed_size,
                                              channels=self._num_filters,
                                              kernel_size=ngram_size,
                                              use_bias=True))
            maxpool_output_size = self._num_filters * len(self._ngram_filter_sizes)
            self._activation = conv_layer_activation
            if self._num_highway:
                self._highways = Highway(maxpool_output_size,
                                         self._num_highway,
                                         activation=self._activation)
            else:
                self._highways = None
            if self._output_size:
                self._projection = nn.Dense(in_units=maxpool_output_size,
                                            units=self._output_size,
                                            use_bias=True)
            else:
                self._projection = None
                self._output_size = maxpool_output_size

    def set_highway_bias(self):
        self._highways.set_bias()

    def forward(self, inputs, mask=None): # pylint: disable=arguments-differ
        r"""
        Forward computation for char_encoder

        Parameters
        ----------
        inputs: NDArray
            The input tensor is of shape `(seq_len, batch_size, embedding_size)` TNC.
        mask: NDArray
            The mask applied to the input of shape `(seq_len, embedding_size)`

        Returns
        ----------
        output: NDArray
            The output of the character encoder with shape `(batch_size, output_Size)`

        """
        if mask is not None:
            inputs = inputs * mask.expand_dims(-1)

        inputs = nd.transpose(inputs, axes=(1, 2, 0))

        filter_outputs = []
        for _, conv in enumerate(self._convs):
            filter_outputs.append(
                self._activation(conv(inputs).max(axis=2))
            )

        output = nd.concat(*filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        if self._highways:
            output = self._highways(output)

        if self._projection:
            output = self._projection(output)

        return output
