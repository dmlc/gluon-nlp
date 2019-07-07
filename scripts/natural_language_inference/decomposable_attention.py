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
# Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>.
# pylint: disable=arguments-differ

"""
Implementation of the decomposable attention model with intra sentence attention.
"""

from mxnet import gluon
from mxnet.gluon import nn


class DecomposableAttentionModel(gluon.HybridBlock):
    """
    A Decomposable Attention Model for Natural Language Inference
    using intra-sentence attention.
    Arxiv paper: https://arxiv.org/pdf/1606.01933.pdf
    """
    def __init__(self, vocab_size, word_embed_size, hidden_size,
                 dropout=0., intra_attention=False, **kwargs):
        super(DecomposableAttentionModel, self).__init__(**kwargs)
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.use_intra_attention = intra_attention
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.word_emb = nn.Embedding(vocab_size, word_embed_size)
            self.lin_proj = nn.Dense(hidden_size, in_units=word_embed_size,
                                     flatten=False, use_bias=False)
            if self.use_intra_attention:
                self.intra_attention = IntraSentenceAttention(hidden_size, hidden_size, dropout)
                input_size = hidden_size * 2
            else:
                self.intra_attention = None
                input_size = hidden_size
            self.model = DecomposableAttention(input_size, hidden_size, 3, dropout)

    def hybrid_forward(self, F, sentence1, sentence2):
        """
        Predict the relation of two sentences.

        Parameters
        ----------
        sentence1 : NDArray
            Shape (batch_size, length)
        sentence2 : NDArray
            Shape (batch_size, length)

        Returns
        -------
        pred : NDArray
            Shape (batch_size, num_classes). num_classes == 3.

        """
        feature1 = self.lin_proj(self.word_emb(sentence1))
        feature2 = self.lin_proj(self.word_emb(sentence2))
        if self.use_intra_attention:
            feature1 = F.concat(feature1, self.intra_attention(feature1), dim=-1)
            feature2 = F.concat(feature2, self.intra_attention(feature2), dim=-1)
        pred = self.model(feature1, feature2)
        return pred

class IntraSentenceAttention(gluon.HybridBlock):
    """
    Intra Sentence Attention block.
    """
    def __init__(self, inp_size, hidden_size, dropout=0., **kwargs):
        super(IntraSentenceAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            # F_intra in the paper
            self.intra_attn_emb = nn.HybridSequential()
            self.intra_attn_emb.add(self.dropout_layer)
            self.intra_attn_emb.add(nn.Dense(hidden_size, in_units=inp_size,
                                             activation='relu', flatten=False))
            self.intra_attn_emb.add(self.dropout_layer)
            self.intra_attn_emb.add(nn.Dense(hidden_size, in_units=hidden_size,
                                             activation='relu', flatten=False))

    def hybrid_forward(self, F, feature_a):
        """
        Compute intra-sentence attention given embedded words.

        Parameters
        ----------
        feature_a : NDArray
            Shape (batch_size, length, hidden_size)

        Returns
        -------
        alpha : NDArray
            Shape (batch_size, length, hidden_size)
        """
        tilde_a = self.intra_attn_emb(feature_a)
        e_matrix = F.batch_dot(tilde_a, tilde_a, transpose_b=True)
        alpha = F.batch_dot(e_matrix.softmax(), tilde_a)
        return alpha

class DecomposableAttention(gluon.HybridBlock):
    """
    Decomposable Attention block.
    """
    def __init__(self, inp_size, hidden_size, num_class, dropout=0., **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            # attention function
            self.f = self._ff_layer(in_units=inp_size, out_units=hidden_size, flatten=False)
            # compare function
            self.g = self._ff_layer(in_units=hidden_size * 2, out_units=hidden_size, flatten=False)
            # predictor
            self.h = self._ff_layer(in_units=hidden_size * 2, out_units=hidden_size, flatten=True)
            self.h.add(nn.Dense(num_class, in_units=hidden_size))
        # extract features
        self.hidden_size = hidden_size
        self.inp_size = inp_size

    def _ff_layer(self, in_units, out_units, flatten=True):
        m = nn.HybridSequential()
        m.add(self.dropout_layer)
        m.add(nn.Dense(out_units, in_units=in_units, activation='relu', flatten=flatten))
        m.add(self.dropout_layer)
        m.add(nn.Dense(out_units, in_units=out_units, activation='relu', flatten=flatten))
        return m

    def hybrid_forward(self, F, a, b):
        """
        Forward of Decomposable Attention layer
        """
        # a.shape = [B, L1, H]
        # b.shape = [B, L2, H]
        # extract features
        tilde_a = self.f(a)  # shape = [B, L1, H]
        tilde_b = self.f(b)  # shape = [B, L2, H]
        # attention
        # e.shape = [B, L1, L2]
        e = F.batch_dot(tilde_a, tilde_b, transpose_b=True)
        # beta: b align to a, [B, L1, H]
        beta = F.batch_dot(e.softmax(), tilde_b)
        # alpha: a align to b, [B, L2, H]
        alpha = F.batch_dot(e.transpose([0, 2, 1]).softmax(), tilde_a)
        # compare
        feature1 = self.g(F.concat(tilde_a, beta, dim=2))
        feature2 = self.g(F.concat(tilde_b, alpha, dim=2))
        feature1 = feature1.sum(axis=1)
        feature2 = feature2.sum(axis=1)
        yhat = self.h(F.concat(feature1, feature2, dim=1))
        return yhat
