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

"""Convolutional encoder."""

from __future__ import absolute_import
from __future__ import print_function

__all__ = ['ConvolutionalEncoder']

from mxnet import gluon
from mxnet.gluon import nn
from gluonnlp.initializer import HighwayBias

from .highway import Highway


class ConvolutionalEncoder(gluon.HybridBlock):
    r"""Convolutional encoder.

    We implement the convolutional encoder proposed in the following work::

        @inproceedings{kim2016character,
         title={Character-Aware Neural Language Models.},
         author={Kim, Yoon and Jernite, Yacine and Sontag, David and Rush, Alexander M},
         booktitle={AAAI},
         pages={2741--2749},
         year={2016}
        }

    Parameters
    ----------
    embed_size : int, default 15
        The input dimension to the encoder.
        We set the default according to the original work's experiments
        on PTB dataset with Char-small model setting.
    num_filters: Tuple[int], default (25, 50, 75, 100, 125, 150)
        The output dimension for each convolutional layer according to the filter sizes,
        which are the number of the filters learned by the layers.
        We set the default according to the original work's experiments
        on PTB dataset with Char-small model setting.
    ngram_filter_sizes: Tuple[int], default (1, 2, 3, 4, 5, 6)
        The size of each convolutional layer,
        and len(ngram_filter_sizes) equals to the number of convolutional layers.
        We set the default according to the original work's experiments
        on PTB dataset with Char-small model setting.
    conv_layer_activation: str, default 'tanh'
        Activation function to be used after convolutional layer.
        We set the default according to the original work's experiments
        on PTB dataset with Char-small model setting.
    num_highway: int, default '1'
        The number of layers of the Highway layer.
        We set the default according to the original work's experiments
        on PTB dataset with Char-small model setting.
    highway_layer_activation: str, default 'relu'
        Activation function to be used after highway layer.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
        We set the default according to the original work's experiments
        on PTB dataset with Char-small model setting.
    highway_bias : HighwayBias,
        default HighwayBias(nonlinear_transform_bias=0.0, transform_gate_bias=-2.0)
        The biases applied to the highway layer.
        We set the default according to the above original work.
    output_size: int, default None
        The output dimension after conducting the convolutions and max pooling,
        and applying highways, as well as linear projection.

    """
    def __init__(self,
                 embed_size=15,
                 num_filters=(25, 50, 75, 100, 125, 150),
                 ngram_filter_sizes=(1, 2, 3, 4, 5, 6),
                 conv_layer_activation='tanh',
                 num_highway=1,
                 highway_layer_activation='relu',
                 highway_bias=HighwayBias(nonlinear_transform_bias=0.0, transform_gate_bias=-2.0),
                 output_size=None,
                 **kwargs):
        super(ConvolutionalEncoder, self).__init__(**kwargs)

        self._embed_size = embed_size
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._num_highway = num_highway
        self._output_size = output_size

        with self.name_scope():
            self._convs = gluon.contrib.nn.HybridConcurrent()
            maxpool_output_size = 0
            with self._convs.name_scope():
                for num_filter, ngram_size in zip(self._num_filters, self._ngram_filter_sizes):
                    seq = nn.HybridSequential()
                    seq.add(nn.Conv1D(in_channels=self._embed_size,
                                      channels=num_filter,
                                      kernel_size=ngram_size,
                                      use_bias=True))
                    seq.add(gluon.nn.HybridLambda(lambda F, x: F.max(x, axis=2)))
                    if conv_layer_activation is not None:
                        seq.add(nn.Activation(conv_layer_activation))
                    self._convs.add(seq)
                    maxpool_output_size += num_filter

            if self._num_highway:
                self._highways = Highway(maxpool_output_size,
                                         self._num_highway,
                                         activation=highway_layer_activation,
                                         highway_bias=highway_bias)
            else:
                self._highways = None
            if self._output_size:
                self._projection = nn.Dense(in_units=maxpool_output_size,
                                            units=self._output_size,
                                            use_bias=True)
            else:
                self._projection = None
                self._output_size = maxpool_output_size

    def hybrid_forward(self, F, inputs, mask=None): # pylint: disable=arguments-differ
        r"""
        Forward computation for char_encoder

        Parameters
        ----------
        inputs: NDArray
            The input tensor is of shape `(seq_len, batch_size, embedding_size)` TNC.
        mask: NDArray
            The mask applied to the input of shape `(seq_len, batch_size)`, the mask will
            be broadcasted along the embedding dimension.

        Returns
        ----------
        output: NDArray
            The output of the encoder with shape `(batch_size, output_size)`

        """
        if mask is not None:
            inputs = F.broadcast_mul(inputs, mask.expand_dims(-1))

        inputs = F.transpose(inputs, axes=(1, 2, 0))

        output = self._convs(inputs)

        if self._highways:
            output = self._highways(output)

        if self._projection:
            output = self._projection(output)

        return output
