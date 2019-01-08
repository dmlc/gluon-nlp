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

r"""
QA encoder of QANet and BiDAF.
"""
import math

from mxnet import gluon, nd
from mxnet.initializer import MSRAPrelu, Xavier
from gluonnlp.model import DotProductAttentionCell, MultiHeadAttentionCell


class QANetEncoder(gluon.HybridBlock):
    r"""
    Stacked block of Embedding encoder or Model encoder.
    """

    def __init__(self, kernel_size, num_filters, layers_dropout, conv_layers=2, num_heads=8,
                 num_blocks=1, **kwargs):
        super(QANetEncoder, self).__init__(**kwargs)

        self._layers_dropout = layers_dropout
        total_layers = float((conv_layers + 2) * num_blocks)
        sub_layer_idx = 1
        self.num_blocks = num_blocks

        with self.name_scope():
            self.qanet_encoder = gluon.nn.HybridSequential()
            with self.qanet_encoder.name_scope():
                for _ in range(num_blocks):
                    self.qanet_encoder.add(
                        QANetEncoderCell(
                            kernel_size=kernel_size,
                            num_filters=num_filters,
                            conv_layers=conv_layers,
                            num_heads=num_heads,
                            total_layers=total_layers,
                            sub_layer_idx=sub_layer_idx
                        )
                    )
                    sub_layer_idx += (conv_layers + 2)

    def hybrid_forward(self, F, x, mask):
        r"""Implement forward computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length, features)`
        mask : NDArray
            input tensor with shape `(batch_size, sequence_length)`

        Returns, NDArray
        --------
            output tensor with shape `(batch_size, sequence_length, features)`
        """
        for encoder in self.qanet_encoder:
            x = encoder(x, mask)
            x = F.Dropout(x, p=self._layers_dropout)
        return x


class QANetEncoderCell(gluon.HybridBlock):
    r"""The basic encoder block.

    Parameters
    ----------
    kernel_size : int
        The kernel size for all depthwise convolution layers.
    num_filters : int
        The number of filters for all convolution layers.
    conv_layers : int
        The number of convolution layers in one encoder block.
    num_heads : int
        The number of heads in multi-head attention layer.
    total_layers : int
    sub_layer_idx : int
        The sub_layer_idx / total_layers is the dropout probability for layer.
    """

    def __init__(self, kernel_size, num_filters, conv_layers, num_heads, total_layers,
                 sub_layer_idx, layers_dropout, p_l, emb_encoder_conv_channels, **kwargs):
        super(QANetEncoderCell, self).__init__(**kwargs)

        self._layers_dropout = layers_dropout
        self._p_l = p_l
        self._emb_encoder_conv_channels = emb_encoder_conv_channels

        with self.name_scope():
            self.position_encoder = PositionalEncoding()
            self.convs = gluon.nn.HybridSequential()
            with self.convs.name_scope():
                for _ in range(conv_layers):
                    conv_layer = gluon.nn.HybridSequential()
                    with conv_layer.name_scope():
                        conv_layer.add(
                            gluon.nn.LayerNorm(epsilon=1e-06)
                        )
                        conv_layer.add(
                            gluon.nn.Dropout(self._layers_dropout)
                        )
                        conv_layer.add(
                            DepthwiseSeparableConvolution(
                                kernel_size=kernel_size,
                                num_filters=num_filters,
                                input_channels=num_filters
                            )
                        )
                        conv_layer.add(
                            StochasticDepthDropoutLayer(
                                dropout=(sub_layer_idx / total_layers) *
                                (1 - self._p_l)
                            )
                        )
                    sub_layer_idx += 1
                    self.convs.add(conv_layer)

            self.dropout = gluon.nn.Dropout(self._layers_dropout)
            self.attention = SelfAttention(num_heads=num_heads)
            self.attention_dropout = StochasticDepthDropoutLayer(
                (sub_layer_idx / total_layers) * (1 - self._p_l))
            sub_layer_idx += 1
            self.attention_layer_norm = gluon.nn.LayerNorm(epsilon=1e-06)

            self.positionwise_ffn = gluon.nn.HybridSequential()
            with self.positionwise_ffn.name_scope():
                self.positionwise_ffn.add(
                    gluon.nn.LayerNorm(epsilon=1e-06)
                )
                self.positionwise_ffn.add(
                    gluon.nn.Dropout(rate=self._layers_dropout)
                )
                self.positionwise_ffn.add(
                    gluon.nn.Dense(
                        units=self._emb_encoder_conv_channels,
                        activation='relu',
                        use_bias=True,
                        weight_initializer=MSRAPrelu(),
                        flatten=False
                    )
                )
                self.positionwise_ffn.add(
                    gluon.nn.Dense(
                        units=self._emb_encoder_conv_channels,
                        use_bias=True,
                        weight_initializer=Xavier(),
                        flatten=False
                    )
                )
                self.positionwise_ffn.add(
                    StochasticDepthDropoutLayer(
                        dropout=(sub_layer_idx / total_layers) * (1 - self._p_l)
                    )
                )

    def hybrid_forward(self, F, x, mask):
        r"""Implement forward computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length, hidden_size)`
        mask : NDArray
            input tensor with shape `(batch_size, sequence_length)`

        Returns
        --------
        x : NDArray
            output tensor with shape `(batch_size, sequence_length, hidden_size)`
        mask : NDArray
            output tensor with shape `(batch_size, sequence_length)`
        """
        x = self.position_encoder(x)
        for conv in self.convs:
            residual = x
            x = conv(x) + residual
        residual = x
        x = self.attention_layer_norm(x)
        x = F.Dropout(x, p=self._layers_dropout)
        x = self.attention(x, mask)
        x = self.attention_dropout(x) + residual
        return x + self.positionwise_ffn(x)


class StochasticDepthDropoutLayer(gluon.HybridBlock):
    r"""
    Stochastic dropout a layer.
    """

    def __init__(self, dropout, **kwargs):
        super(StochasticDepthDropoutLayer, self).__init__(**kwargs)

        self.dropout = dropout

        with self.name_scope():
            self.dropout_layer = gluon.nn.Dropout(dropout)

    def hybrid_forward(self, F, inputs):
        if F.random.uniform().asscalar() < self.dropout:
            return F.zeros(shape=(1,))
        else:
            return self.dropout_layer(inputs)


class SelfAttention(gluon.HybridBlock):
    r"""
    Implementation of self-attention with gluonnlp.model.MultiHeadAttentionCell
    """

    def __init__(self, num_heads, layers_dropout, emb_encoder_conv_channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

        self._layers_dropout = layers_dropout
        self._emb_encoder_conv_channels = emb_encoder_conv_channels

        with self.name_scope():
            self.attention = MultiHeadAttentionCell(
                num_heads=num_heads,
                base_cell=DotProductAttentionCell(
                    scaled=True,
                    dropout=self._layers_dropout,
                    use_bias=False
                ),
                query_units=self._emb_encoder_conv_channels,
                key_units=self._emb_encoder_conv_channels,
                value_units=self._emb_encoder_conv_channels,
                use_bias=False,
                weight_initializer=Xavier()
            )

    def hybrid_forward(self, F, x, mask):
        r"""Implement forward computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length, hidden_size)`
        mask : NDArray
            input tensor with shape `(batch_size, sequence_length)`

        Returns
        --------
        x : NDArray
            output tensor with shape `(batch_size, sequence_length, hidden_size)`
        """
        mask = F.batch_dot(mask.expand_dims(axis=2), mask.expand_dims(axis=1))
        return self.attention(x, x, mask=mask)[0]


class PositionalEncoding(gluon.HybridBlock):
    r"""
    An implementation of position encoder.
    """

    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)


    def hybrid_forward(self, F, x, min_timescale=1.0, max_timescale=1e4):
        r"""Implement forward computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length, hidden_size)`

        Returns
        --------
         : NDArray
            output tensor with shape `(batch_size, sequence_length, hidden_size)`
        """
        length = x.shape[1]
        channels = x.shape[2]
        position = nd.array(range(length))
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))
        inv_timescales = min_timescale * \
            nd.exp(nd.array(range(num_timescales)) * -log_timescale_increment)
        scaled_time = F.expand_dims(
            position, 1) * F.expand_dims(inv_timescales, 0)
        signal = F.concat(F.sin(scaled_time), F.cos(scaled_time), dim=1)
        signal = F.reshape(signal, shape=(1, length, channels))
        return x + signal.as_in_context(x.context)


class DepthwiseSeparableConvolution(gluon.HybridBlock):
    r"""
    An implementation of depthwise-convolution net.
    """

    def __init__(self, kernel_size, num_filters, input_channels, **kwargs):
        super(DepthwiseSeparableConvolution, self).__init__(**kwargs)
        with self.name_scope():
            self.depthwise_conv = gluon.nn.Conv1D(
                channels=input_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=input_channels,
                use_bias=False,
                weight_initializer=MSRAPrelu()
            )
            self.pointwise_conv = gluon.nn.Conv1D(
                channels=num_filters,
                kernel_size=1,
                activation='relu',
                use_bias=True,
                weight_initializer=MSRAPrelu(),
                bias_initializer='zeros'
            )

    def hybrid_forward(self, F, inputs):
        r"""Implement forward computation.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(batch_size, sequence_length, hidden_size)`

        Returns
        --------
        x : NDArray
            output tensor with shape `(batch_size, sequence_length, new_hidden_size)`
        """
        tmp = F.transpose(inputs, axes=(0, 2, 1))
        depthwise_conv = self.depthwise_conv(tmp)
        outputs = self.pointwise_conv(depthwise_conv)
        return F.transpose(outputs, axes=(0, 2, 1))
