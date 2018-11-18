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

# pylint: disable=
r"""
This file contains QANet model and all used layers.
"""
import math

import mxnet as mx
from mxnet import gluon, nd
from mxnet.initializer import MSRAPrelu, Normal, Uniform, Xavier
from gluonnlp.initializer import HighwayBias
from gluonnlp.model import (ConvolutionalEncoder, DotProductAttentionCell,
                            Highway, MultiHeadAttentionCell)

from config import opt
from util import mask_logits


class MySoftmaxCrossEntropy(gluon.loss.Loss):
    r"""Caluate the sum of softmax cross entropy.

    Reference:
    http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probalbility distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead of
        unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """

    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None, batch_axis=0,
                 **kwargs):
        super(MySoftmaxCrossEntropy, self).__init__(
            weight, batch_axis, **kwargs)
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=axis,
                                                       sparse_label=sparse_label,
                                                       from_logits=from_logits,
                                                       weight=weight,
                                                       batch_axis=batch_axis)

    def forward(self, predict_begin, predict_end, label_begin, label_end):
        r"""Implement forward computation.

        Parameters
        -----------
        predict_begin : NDArray
            Predicted probability distribution of answer begin position,
            input tensor with shape `(batch_size, sequence_length)`
        predict_end : NDArray
            Predicted probability distribution of answer end position,
            input tensor with shape `(batch_size, sequence_length)`
        label_begin : NDArray
            True label of the answer begin position,
            input tensor with shape `(batch_size, )`
        label_end : NDArray
            True label of the answer end position,
            input tensor with shape `(batch_size, )`

        Returns
        --------
        out: NDArray
            output tensor with shape `(batch_size, )`
        """
        return self.loss(predict_begin, label_begin) + self.loss(predict_end, label_end)


class QANet(gluon.HybridBlock):
    r"""QANet model.
    We implemented the QANet proposed in the following work::
        @article{DBLP:journals/corr/abs-1804-09541,
            author    = {Adams Wei Yu and
                        David Dohan and
                        Minh{-}Thang Luong and
                        Rui Zhao and
                        Kai Chen and
                        Mohammad Norouzi and
                        Quoc V. Le},
            title     = {QANet: Combining Local Convolution with Global Self-Attention for
                        Reading Comprehension},
            year      = {2018},
            url       = {http://arxiv.org/abs/1804.09541}
        }

    """

    def __init__(self, **kwargs):
        super(QANet, self).__init__(**kwargs)
        with self.name_scope():
            self.flatten = gluon.nn.Flatten()
            self.dropout = gluon.nn.Dropout(opt.layers_dropout)
            self.char_conv = ConvolutionalEncoder(
                embed_size=opt.char_emb_dim,
                num_filters=opt.char_conv_filters,
                ngram_filter_sizes=opt.char_conv_ngrams,
                conv_layer_activation=None,
                num_highway=0
            )

        self.highway = gluon.nn.HybridSequential()
        with self.highway.name_scope():
            self.highway.add(
                gluon.nn.Dense(
                    units=opt.emb_encoder_conv_channels,
                    flatten=False,
                    use_bias=False,
                    weight_initializer=Xavier()
                )
            )
            self.highway.add(
                Highway(
                    input_size=opt.emb_encoder_conv_channels,
                    num_layers=opt.highway_layers,
                    activation='relu',
                    highway_bias=HighwayBias(
                        nonlinear_transform_bias=0.0,
                        transform_gate_bias=0.0
                    )
                )
            )

        self.word_emb = gluon.nn.HybridSequential()
        with self.word_emb.name_scope():
            self.word_emb.add(
                gluon.nn.Embedding(
                    input_dim=opt.word_corpus,
                    output_dim=opt.word_emb_dim
                )
            )
            self.word_emb.add(
                gluon.nn.Dropout(rate=opt.word_emb_dropout)
            )
        self.char_emb = gluon.nn.HybridSequential()
        with self.char_emb.name_scope():
            self.char_emb.add(
                gluon.nn.Embedding(
                    input_dim=opt.character_corpus,
                    output_dim=opt.char_emb_dim,
                    weight_initializer=Normal(sigma=0.1)
                )
            )
            self.char_emb.add(
                gluon.nn.Dropout(rate=opt.char_emb_dropout)
            )

        with self.name_scope():
            self.emb_encoder = Encoder(
                kernel_size=opt.emb_encoder_conv_kernerl_size,
                num_filters=opt.emb_encoder_conv_channels,
                conv_layers=opt.emb_encoder_num_conv_layers,
                num_heads=opt.emb_encoder_num_head,
                num_blocks=opt.emb_encoder_num_block
            )

            self.project = gluon.nn.Dense(
                units=opt.emb_encoder_conv_channels,
                flatten=False,
                use_bias=False,
                weight_initializer=Xavier()
            )

        with self.name_scope():
            self.co_attention = CoAttention()

        with self.name_scope():
            self.model_encoder = Encoder(
                kernel_size=opt.model_encoder_conv_kernel_size,
                num_filters=opt.model_encoder_conv_channels,
                conv_layers=opt.model_encoder_conv_layers,
                num_heads=opt.model_encoder_num_head,
                num_blocks=opt.model_encoder_num_block
            )

        with self.name_scope():
            self.predict_begin = gluon.nn.Dense(
                units=1,
                use_bias=True,
                flatten=False,
                weight_initializer=Xavier(
                    rnd_type='uniform', factor_type='in', magnitude=1),
                bias_initializer=Uniform(1.0/opt.model_encoder_conv_channels)
            )
            self.predict_end = gluon.nn.Dense(
                units=1,
                use_bias=True,
                flatten=False,
                weight_initializer=Xavier(
                    rnd_type='uniform', factor_type='in', magnitude=1),
                bias_initializer=Uniform(1.0/opt.model_encoder_conv_channels)
            )

    def hybrid_forward(self, F, context, query, context_char, query_char,
                       y_begin, y_end):
        r"""Implement forward computation.

        Parameters
        -----------
        context : NDArray
            input tensor with shape `(batch_size, context_sequence_length)`
        query : NDArray
            input tensor with shape `(batch_size, query_sequence_length)`
        context_char : NDArray
            input tensor with shape `(batch_size, context_sequence_length, num_char_per_word)`
        query_char : NDArray
            input tensor with shape `(batch_size, query_sequence_length, num_char_per_word)`
        y_begin : NDArray
            input tensor with shape `(batch_size, )`
        y_end : NDArray
            input tensor with shape `(batch_size, )`

        Returns
        --------
        predicted_begin : NDArray
            output tensor with shape `(batch_size, context_sequence_length)`
        predicted_end : NDArray
            output tensor with shape `(batch_size, context_sequence_length)`
        """
        (batch, _) = context.shape
        context_mask = context > 0
        query_mask = query > 0
        context_max_len = int(context_mask.sum(axis=1).max().asscalar())
        query_max_len = int(query_mask.sum(axis=1).max().asscalar())

        context = F.slice(context, begin=(0, 0), end=(batch, context_max_len))
        query = F.slice(query, begin=(0, 0), end=(batch, query_max_len))
        context_mask = F.slice(
            context_mask,
            begin=(0, 0),
            end=(batch, context_max_len)
        )
        query_mask = F.slice(
            query_mask,
            begin=(0, 0),
            end=(batch, query_max_len)
        )
        context_char = F.slice(
            context_char,
            begin=(0, 0, 0),
            end=(batch, context_max_len, opt.max_character_per_word)
        )
        query_char = F.slice(
            query_char,
            begin=(0, 0, 0),
            end=(batch, query_max_len, opt.max_character_per_word)
        )

        # word embedding
        context_word_emb = self.word_emb(context)
        query_word_emb = self.word_emb(query)

        # char embedding
        context_char_flat = self.flatten(context_char)
        query_char_flat = self.flatten(query_char)
        context_char_emb = self.char_emb(context_char_flat)
        query_char_emb = self.char_emb(query_char_flat)

        context_char_emb = F.reshape(
            context_char_emb,
            shape=(
                batch*context_max_len,
                opt.max_character_per_word,
                opt.char_emb_dim
            )
        )
        query_char_emb = F.reshape(
            query_char_emb,
            shape=(
                batch*query_max_len,
                opt.max_character_per_word,
                opt.char_emb_dim
            )
        )
        context_char_emb = F.transpose(context_char_emb, axes=(1, 0, 2))
        query_char_emb = F.transpose(query_char_emb, axes=(1, 0, 2))
        context_char_emb = self.char_conv(context_char_emb)
        query_char_emb = self.char_conv(query_char_emb)
        context_char_emb = F.reshape(
            context_char_emb,
            shape=(
                batch,
                context_max_len,
                context_char_emb.shape[-1]
            )
        )
        query_char_emb = F.reshape(
            query_char_emb,
            shape=(
                batch,
                query_max_len,
                query_char_emb.shape[-1]
            )
        )

        # concat word and char embedding
        context_concat = F.concat(context_word_emb, context_char_emb, dim=-1)
        query_concat = F.concat(query_word_emb, query_char_emb, dim=-1)

        # highway net
        context_final_emb = self.highway(context_concat)
        query_final_emb = self.highway(query_concat)

        # embedding encoder
        # share the weights between passage and question
        context_emb_encoded = self.emb_encoder(context_final_emb, context_mask)
        query_emb_encoded = self.emb_encoder(query_final_emb, query_mask)

        # context-query attention layer
        M = self.co_attention(context_emb_encoded, query_emb_encoded, context_mask,
                              query_mask, context_max_len, query_max_len)

        M = self.project(M)
        M = self.dropout(M)

        # model encoder layer
        M_0 = self.model_encoder(M, context_mask)
        M_1 = self.model_encoder(M_0, context_mask)
        M_2 = self.model_encoder(M_1, context_mask)

        # predict layer
        begin_hat = self.flatten(
            self.predict_begin(F.concat(M_0, M_1, dim=-1)))
        end_hat = self.flatten(self.predict_end(F.concat(M_0, M_2, dim=-1)))
        predicted_begin = mask_logits(begin_hat, context_mask)
        predicted_end = mask_logits(end_hat, context_mask)
        return predicted_begin, predicted_end, y_begin, y_end


class Encoder(gluon.HybridBlock):
    r"""
    Stacked block of Embedding encoder or Model encoder.
    """

    def __init__(self, kernel_size, num_filters, conv_layers=2, num_heads=8,
                 num_blocks=1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dropout = gluon.nn.Dropout(opt.layers_dropout)
        total_layers = float((conv_layers + 2) * num_blocks)
        sub_layer_idx = 1
        self.num_blocks = num_blocks
        self.stack_encoders = gluon.nn.HybridSequential()
        with self.stack_encoders.name_scope():
            for _ in range(num_blocks):
                self.stack_encoders.add(
                    OneEncoderBlock(
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
        for encoder in self.stack_encoders:
            x = encoder(x, mask)
            x = F.Dropout(x, p=opt.layers_dropout)
        return x


class OneEncoderBlock(gluon.HybridBlock):
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
                 sub_layer_idx, **kwargs):
        super(OneEncoderBlock, self).__init__(**kwargs)
        self.position_encoder = PositionEncoder()
        self.convs = gluon.nn.HybridSequential()
        with self.convs.name_scope():
            for _ in range(conv_layers):
                one_conv_module = gluon.nn.HybridSequential()
                with one_conv_module.name_scope():
                    one_conv_module.add(
                        gluon.nn.LayerNorm(epsilon=1e-06)
                    )
                    one_conv_module.add(
                        gluon.nn.Dropout(opt.layers_dropout)
                    )
                    one_conv_module.add(
                        DepthwiseConv(
                            kernel_size=kernel_size,
                            num_filters=num_filters,
                            input_channels=num_filters
                        )
                    )
                    one_conv_module.add(
                        StochasticDropoutLayer(
                            dropout=(sub_layer_idx / total_layers) *
                            (1 - opt.p_l)
                        )
                    )
                sub_layer_idx += 1
                self.convs.add(one_conv_module)

        with self.name_scope():
            self.dropout = gluon.nn.Dropout(opt.layers_dropout)
            self.attention = SelfAttention(num_heads=num_heads)
            self.attention_dropout = StochasticDropoutLayer(
                (sub_layer_idx / total_layers) * (1 - opt.p_l))
            sub_layer_idx += 1
            self.attention_layer_norm = gluon.nn.LayerNorm(epsilon=1e-06)

        self.positionwise_ffn = gluon.nn.HybridSequential()
        with self.positionwise_ffn.name_scope():
            self.positionwise_ffn.add(
                gluon.nn.LayerNorm(epsilon=1e-06)
            )
            self.positionwise_ffn.add(
                gluon.nn.Dropout(rate=opt.layers_dropout)
            )
            self.positionwise_ffn.add(
                gluon.nn.Dense(
                    units=opt.emb_encoder_conv_channels,
                    activation='relu',
                    use_bias=True,
                    weight_initializer=MSRAPrelu(),
                    flatten=False
                )
            )
            self.positionwise_ffn.add(
                gluon.nn.Dense(
                    units=opt.emb_encoder_conv_channels,
                    use_bias=True,
                    weight_initializer=Xavier(),
                    flatten=False
                )
            )
            self.positionwise_ffn.add(
                StochasticDropoutLayer(
                    dropout=(sub_layer_idx / total_layers) * (1 - opt.p_l)
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
        x = F.Dropout(x, p=opt.layers_dropout)
        x = self.attention(x, mask)
        x = self.attention_dropout(x) + residual
        return x + self.positionwise_ffn(x)


class StochasticDropoutLayer(gluon.HybridBlock):
    r"""
    Stochastic dropout a layer.
    """

    def __init__(self, dropout, **kwargs):
        super(StochasticDropoutLayer, self).__init__(**kwargs)
        self.dropout = dropout
        with self.name_scope():
            self.dropout_fn = gluon.nn.Dropout(dropout)

    def hybrid_forward(self, F, inputs):
        if F.random.uniform().asscalar() < self.dropout:
            return F.zeros(shape=(1,))
        else:
            return self.dropout_fn(inputs)


class SelfAttention(gluon.HybridBlock):
    r"""
    Implementation of self-attention with gluonnlp.model.MultiHeadAttentionCell
    """

    def __init__(self, num_heads, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.attention = MultiHeadAttentionCell(
                num_heads=num_heads,
                base_cell=DotProductAttentionCell(
                    scaled=True,
                    dropout=opt.layers_dropout,
                    use_bias=False
                ),
                query_units=opt.emb_encoder_conv_channels,
                key_units=opt.emb_encoder_conv_channels,
                value_units=opt.emb_encoder_conv_channels,
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


class PositionEncoder(gluon.HybridBlock):
    r"""
    An implementation of position encoder.
    """

    def __init__(self, **kwargs):
        super(PositionEncoder, self).__init__(**kwargs)
        with self.name_scope():
            pass

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


class DepthwiseConv(gluon.HybridBlock):
    r"""
    An implementation of depthwise-convolution net.
    """

    def __init__(self, kernel_size, num_filters, input_channels, **kwargs):
        super(DepthwiseConv, self).__init__(**kwargs)
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


class CoAttention(gluon.HybridBlock):
    r"""
    An implementation of co-attention block.
    """

    def __init__(self, **kwargs):
        super(CoAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.w4c = gluon.nn.Dense(
                units=1,
                flatten=False,
                weight_initializer=Xavier(),
                use_bias=False
            )
            self.w4q = gluon.nn.Dense(
                units=1,
                flatten=False,
                weight_initializer=Xavier(),
                use_bias=False
            )
            self.w4mlu = self.params.get(
                'linear_kernel', shape=(1, 1, opt.emb_encoder_conv_channels), init=mx.init.Xavier())
            self.bias = self.params.get(
                'coattention_bias', shape=(1,), init=mx.init.Zero())

    def hybrid_forward(self, F, context, query, context_mask, query_mask,
                       context_max_len, query_max_len, w4mlu, bias):
        """Implement forward computation.

        Parameters
        -----------
        context : NDArray
            input tensor with shape `(batch_size, context_sequence_length, hidden_size)`
        query : NDArray
            input tensor with shape `(batch_size, query_sequence_length, hidden_size)`
        context_mask : NDArray
            input tensor with shape `(batch_size, context_sequence_length)`
        query_mask : NDArray
            input tensor with shape `(batch_size, query_sequence_length)`
        context_max_len : int
        query_max_len : int

        Returns
        --------
        return : NDArray
            output tensor with shape `(batch_size, context_sequence_length, 4*hidden_size)`
        """
        context_mask = F.expand_dims(context_mask, axis=-1)
        query_mask = F.expand_dims(query_mask, axis=1)

        similarity = self._calculate_trilinear_similarity(
            context, query, context_max_len, query_max_len, w4mlu, bias)

        similarity_dash = F.softmax(mask_logits(similarity, query_mask))
        similarity_dash_trans = F.transpose(F.softmax(
            mask_logits(similarity, context_mask), axis=1), axes=(0, 2, 1))
        c2q = F.batch_dot(similarity_dash, query)
        q2c = F.batch_dot(F.batch_dot(
            similarity_dash, similarity_dash_trans), context)
        return F.concat(context, c2q, context * c2q, context * q2c, dim=-1)

    def _calculate_trilinear_similarity(self, context, query, context_max_len, query_max_len,
                                        w4mlu, bias):
        """Implement the computation of trilinear similarity function.

            refer https://github.com/NLPLearn/QANet/blob/master/layers.py#L505

            The similarity function is:
                    f(w, q) = W[w, q, w * q]
            where w and q represent the word in context and query respectively,
            and * operator means hadamard product.

        Parameters
        -----------
        context : NDArray
            input tensor with shape `(batch_size, context_sequence_length, hidden_size)`
        query : NDArray
            input tensor with shape `(batch_size, query_sequence_length, hidden_size)`
        context_max_len : int
        context_max_len : int

        Returns
        --------
        similarity_mat : NDArray
            output tensor with shape `(batch_size, context_sequence_length, query_sequence_length)`
        """

        subres0 = nd.tile(self.w4c(context), [1, 1, query_max_len])
        subres1 = nd.tile(nd.transpose(
            self.w4q(query), axes=(0, 2, 1)), [1, context_max_len, 1])
        subres2 = nd.batch_dot(w4mlu * context,
                               nd.transpose(query, axes=(0, 2, 1)))
        similarity_mat = subres0 + subres1 + subres2 + bias
        return similarity_mat
