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

# A Character Convolution + BiLSTM + CRF network for NER

# @author：kenjewu
# @date：2018/12/12


import mxnet as mx
import gluonnlp as nlp
from mxnet import nd
from mxnet.gluon import nn, rnn
from .crf import CRF


class CNN_BILSTM_CRF(nn.Block):
    def __init__(self, nchars, nchar_embed, nwords, nword_embed, nfilters, kernel_size,
                 nhiddens, nlayers, ntag_space, emb_drop_prob, rnn_drop_prob, out_drop_prob,
                 tag2idx, **kwargs):
        super(CNN_BILSTM_CRF, self).__init__(**kwargs)
        with self.name_scope():
            self.kernel_size = kernel_size
            self.char_embedding_layer = nn.Embedding(nchars, nchar_embed)
            self.word_embedding_layer = nn.Embedding(nwords, nword_embed)

            self.char_conv_encoder = nlp.model.ConvolutionalEncoder(embed_size=nchar_embed,
                                                                    num_filters=(nfilters,),
                                                                    ngram_filter_sizes=(kernel_size,),
                                                                    conv_layer_activation=None,
                                                                    num_highway=0)

            self.bi_lstm_layer = rnn.LSTM(nhiddens, num_layers=nlayers,
                                          bidirectional=True, dropout=rnn_drop_prob[1])

            self.tag_dense_layer = None
            if ntag_space:
                self.tag_dense_layer = nn.Dense(ntag_space, flatten=False)
            self.elu_act_layer = nn.ELU()

            self.projection_layer = nn.Dense(len(tag2idx), flatten=False)

            self.crf_layer = CRF(tag2idx)

            self.emb_dropout_layer = nn.Dropout(emb_drop_prob)
            self.rnn_dropout_layer = nn.Dropout(rnn_drop_prob[0])
            self.out_dropout_layer = nn.Dropout(out_drop_prob)

    def _get_rnn_output(self, char_idx, word_idx, state, mask=None, length=None):
        if length is None and mask is not None:
            length = mask.sum(axis=1)

        # (batch_size, max_seq_len, nword_embed)
        word_embed = self.word_embedding_layer(word_idx)

        # (batch_size, max_seq_len, char_len, nchar_embed)
        char_embed = self.char_embedding_layer(char_idx)
        template = nd.zeros_like(char_embed)

        # first reshape to (batch_size*max_seq_len, char_len, nchar_embed)
        char_embed = nd.reshape(char_embed, (-3, -2))
        # then transope to (char_len, batch_size*max_seq_len, nchar_embed)
        char_embed = nd.transpose(char_embed, axes=(1, 0, 2))
        # (batch_size*max_seq_len, nfilters)
        char_embed = self.char_conv_encoder(char_embed)
        # (batch_size, max_seq_len, nfilters)
        char_embed = nd.reshape_like(char_embed, template, lhs_begin=0,
                                     lhs_end=1, rhs_begin=0, rhs_end=2)

        # apply dropout word on input
        word_embed = self.emb_dropout_layer(word_embed)
        char_embed = self.emb_dropout_layer(char_embed)

        # (batch_size, max_seq_len, nword_embed+nfilters)
        lstm_input = nd.concat(word_embed, char_embed, dim=2)
        # (max_seq_len, batch_size, nword_embed+nfilters)
        lstm_input = nd.transpose(lstm_input, axes=(1, 0, 2))
        # apply dropout rnn input
        lstm_input = self.rnn_dropout_layer(lstm_input)

        if length is not None:
            raise NotImplementedError
        else:
            lstm_output, state = self.bi_lstm_layer(lstm_input, state)

        # apply dropout for the output of rnn
        output = self.out_dropout_layer(lstm_output)

        if self.tag_dense_layer is not None:
            output = self.out_dropout_layer(self.elu_act_layer(self.tag_dense_layer(output)))

        return output, state, mask, length

    def forward(self, char_idx, word_idx, state, mask=None, length=None):
        rnn_output, state, mask, length = self._get_rnn_output(char_idx, word_idx,
                                                               state, mask, length)
        feats = self.projection_layer(rnn_output)

        # crf is not hybrid
        score, tag_seq = self.crf_layer(feats)

        return score, tag_seq, feats, state

    def begin_state(self, *args, **kwargs):
        return self.bi_lstm_layer.begin_state(*args, **kwargs)
