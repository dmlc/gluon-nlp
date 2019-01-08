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
QA attention layers for QANet and BiDAF.
"""

from mxnet import gluon, nd
from mxnet.initializer import Xavier, Zero

from .utils import mask_logits


class ContextQueryAttention(gluon.HybridBlock):
    r"""
    An implementation of co-attention block.
    """

    def __init__(self, emb_encoder_conv_channels=128, **kwargs):
        super(ContextQueryAttention, self).__init__(**kwargs)

        self._emb_encoder_conv_channels = emb_encoder_conv_channels

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
                'linear_kernel', shape=(1, 1, self._emb_encoder_conv_channels), init=Xavier())
            self.bias = self.params.get(
                'contextqueryattention_bias', shape=(1,), init=Zero())

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
