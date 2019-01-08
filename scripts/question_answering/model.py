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
QANet model.
"""

from mxnet import gluon
from mxnet.initializer import Normal, Uniform, Xavier
from gluonnlp.initializer import HighwayBias
from gluonnlp.model import ConvolutionalEncoder, Highway

from .qa_encoder import QANetEncoder
from .qa_attention import ContextQueryAttention
from .utils import mask_logits


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

    def __init__(self, word_emb_dim, char_emb_dim, character_corpus, word_corpus, highway_layers,
                 char_conv_filters, char_conv_ngrams, emb_encoder_conv_channels,
                 emb_encoder_conv_kernerl_size, emb_encoder_num_conv_layers, emb_encoder_num_head,
                 emb_encoder_num_block, model_encoder_conv_channels, model_encoder_conv_kernel_size,
                 model_encoder_conv_layers, model_encoder_num_head, model_encoder_num_block,
                 layers_dropout, p_l, word_emb_dropout, char_emb_dropout,
                 max_context_sentence_len, max_question_sentence_len,
                 max_character_per_word, **kwargs):
        super(QANet, self).__init__(**kwargs)

        self._word_emb_dim = word_emb_dim
        self._char_emb_dim = char_emb_dim
        self._character_corpus = character_corpus
        self._word_corpus = word_corpus
        self._highway_layers = highway_layers
        self._char_conv_filters = char_conv_filters
        self._char_conv_ngrams = char_conv_ngrams
        self._emb_encoder_conv_channels = emb_encoder_conv_channels
        self._emb_encoder_conv_kernerl_size = emb_encoder_conv_kernerl_size
        self._emb_encoder_num_conv_layers = emb_encoder_num_conv_layers
        self._emb_encoder_num_head = emb_encoder_num_head
        self._emb_encoder_num_block = emb_encoder_num_block
        self._model_encoder_conv_channels = model_encoder_conv_channels
        self._model_encoder_conv_kernel_size = model_encoder_conv_kernel_size
        self._model_encoder_conv_layers = model_encoder_conv_layers
        self._model_encoder_num_head = model_encoder_num_head
        self._model_encoder_num_block = model_encoder_num_block
        self._layers_dropout = layers_dropout
        self._p_l = p_l
        self._word_emb_dropout = word_emb_dropout
        self._char_emb_dropout = char_emb_dropout
        self._max_context_sentence_len = max_context_sentence_len
        self._max_question_sentence_len = max_question_sentence_len
        self._max_character_per_word = max_character_per_word

        with self.name_scope():

            self.flatten = gluon.nn.Flatten()
            self.dropout = gluon.nn.Dropout(self._layers_dropout)
            self.char_conv = ConvolutionalEncoder(
                embed_size=self._char_emb_dim,
                num_filters=self._char_conv_filters,
                ngram_filter_sizes=self._char_conv_ngrams,
                conv_layer_activation=None,
                num_highway=0
            )

            self.highway = gluon.nn.HybridSequential()
            with self.highway.name_scope():
                self.highway.add(
                    gluon.nn.Dense(
                        units=self._emb_encoder_conv_channels,
                        flatten=False,
                        use_bias=False,
                        weight_initializer=Xavier()
                    )
                )
                self.highway.add(
                    Highway(
                        input_size=self._emb_encoder_conv_channels,
                        num_layers=self._highway_layers,
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
                        input_dim=self._word_corpus,
                        output_dim=self._word_emb_dim
                    )
                )
                self.word_emb.add(
                    gluon.nn.Dropout(rate=self._word_emb_dropout)
                )
            self.char_emb = gluon.nn.HybridSequential()
            with self.char_emb.name_scope():
                self.char_emb.add(
                    gluon.nn.Embedding(
                        input_dim=self._character_corpus,
                        output_dim=self._char_emb_dim,
                        weight_initializer=Normal(sigma=0.1)
                    )
                )
                self.char_emb.add(
                    gluon.nn.Dropout(rate=self._char_emb_dropout)
                )

            self.emb_encoder = QANetEncoder(
                kernel_size=self._emb_encoder_conv_kernerl_size,
                num_filters=self._emb_encoder_conv_channels,
                conv_layers=self._emb_encoder_num_conv_layers,
                num_heads=self._emb_encoder_num_head,
                num_blocks=self._emb_encoder_num_block
            )

            self.project = gluon.nn.Dense(
                units=self._emb_encoder_conv_channels,
                flatten=False,
                use_bias=False,
                weight_initializer=Xavier()
            )

            self.context_query_attention = ContextQueryAttention()

            self.model_encoder = QANetEncoder(
                kernel_size=self._model_encoder_conv_kernel_size,
                num_filters=self._model_encoder_conv_channels,
                conv_layers=self._model_encoder_conv_layers,
                num_heads=self._model_encoder_num_head,
                num_blocks=self._model_encoder_num_block
            )

            self.predict_begin = gluon.nn.Dense(
                units=1,
                use_bias=True,
                flatten=False,
                weight_initializer=Xavier(
                    rnd_type='uniform', factor_type='in', magnitude=1),
                bias_initializer=Uniform(1.0/self._model_encoder_conv_channels)
            )
            self.predict_end = gluon.nn.Dense(
                units=1,
                use_bias=True,
                flatten=False,
                weight_initializer=Xavier(
                    rnd_type='uniform', factor_type='in', magnitude=1),
                bias_initializer=Uniform(1.0/self._model_encoder_conv_channels)
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
            end=(batch, context_max_len, self._max_character_per_word)
        )
        query_char = F.slice(
            query_char,
            begin=(0, 0, 0),
            end=(batch, query_max_len, self._max_character_per_word)
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
                self._max_character_per_word,
                self._char_emb_dim
            )
        )
        query_char_emb = F.reshape(
            query_char_emb,
            shape=(
                batch*query_max_len,
                self._max_character_per_word,
                self._char_emb_dim
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
        M = self.context_query_attention(context_emb_encoded, query_emb_encoded, context_mask,
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
