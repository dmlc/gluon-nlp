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

"""BiDAF model blocks"""

__all__ = ['BiDAFEmbedding', 'BiDAFModelingLayer', 'BiDAFOutputLayer', 'BiDAFModel']

from mxnet import initializer
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.rnn import LSTM

from gluonnlp.model import ConvolutionalEncoder, Highway

from .attention_flow import AttentionFlow
from .bidaf import BidirectionalAttentionFlow
from .similarity_function import LinearSimilarity
from .utils import get_very_negative_number


class BiDAFEmbedding(HybridBlock):
    """BiDAFEmbedding is a class describing embeddings that are separately applied to question
    and context of the datasource. Both question and context are passed in two NDArrays:
    1. Matrix of words: batch_size x words_per_question/context
    2. Tensor of characters: batch_size x words_per_question/context x chars_per_word
    """
    def __init__(self, batch_size, word_vocab, char_vocab, max_seq_len,
                 contextual_embedding_nlayers=2, highway_nlayers=2, embedding_size=100,
                 dropout=0.2, prefix=None, params=None):
        super(BiDAFEmbedding, self).__init__(prefix=prefix, params=params)

        self._word_vocab = word_vocab
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._embedding_size = embedding_size

        with self.name_scope():
            self._char_dense_embedding = nn.Embedding(input_dim=len(char_vocab),
                                                      output_dim=8,
                                                      weight_initializer=initializer.Uniform(0.001))
            self._dropout = nn.Dropout(rate=dropout)
            self._char_conv_embedding = ConvolutionalEncoder(
                embed_size=8,
                num_filters=(100,),
                ngram_filter_sizes=(5,),
                num_highway=None,
                conv_layer_activation='relu',
                output_size=None
            )

            self._word_embedding = nn.Embedding(prefix='predefined_embedding_layer',
                                                input_dim=len(word_vocab),
                                                output_dim=embedding_size)

            self._highway_network = Highway(2 * embedding_size, num_layers=highway_nlayers)
            self._contextual_embedding = LSTM(hidden_size=embedding_size,
                                              num_layers=contextual_embedding_nlayers,
                                              bidirectional=True, input_size=2 * embedding_size,
                                              dropout=dropout)

    def init_embeddings(self, grad_req='null'):
        """Initialize words embeddings with provided embedding values

        Parameters
        ----------
        grad_req: str
            How to treat gradients of embedding layer
        """
        self._word_embedding.weight.set_data(self._word_vocab.embedding.idx_to_vec)
        self._word_embedding.collect_params().setattr('grad_req', grad_req)

    def hybrid_forward(self, F, w, c, *args):  # pylint: disable=arguments-differ
        word_embedded = self._word_embedding(w)
        char_level_data = self._char_dense_embedding(c)
        char_level_data = self._dropout(char_level_data)

        # Transpose to put seq_len first axis to iterate over it
        char_level_data = F.transpose(char_level_data, axes=(1, 2, 0, 3))

        def convolute(token_of_all_batches, _):
            return self._char_conv_embedding(token_of_all_batches), []
        char_embedded, _ = F.contrib.foreach(convolute, char_level_data, [])

        # Transpose to TNC, to join with character embedding
        word_embedded = F.transpose(word_embedded, axes=(1, 0, 2))
        highway_input = F.concat(char_embedded, word_embedded, dim=2)

        def highway(token_of_all_batches, _):
            return self._highway_network(token_of_all_batches), []

        # Pass through highway, shape remains unchanged
        highway_output, _ = F.contrib.foreach(highway, highway_input, [])

        # Transpose to TNC - default for LSTM
        ce_output = self._contextual_embedding(highway_output)
        return ce_output


class BiDAFModelingLayer(HybridBlock):
    """BiDAFModelingLayer implements modeling layer of BiDAF paper. It is used to scan over context
    produced by Attentional Flow Layer via 2 layer bi-LSTM.

    The input data for the forward should be of dimension 8 * hidden_size (default hidden_size
    is 100).

    Parameters
    ----------

    input_dim : `int`, default 100
        The number of features in the hidden state h of LSTM
    nlayers : `int`, default 2
        Number of recurrent layers.
    biflag: `bool`, default True
        If `True`, becomes a bidirectional RNN.
    dropout: `float`, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer.
    prefix : `str` or None
        Prefix of this `Block`.
    params : `ParameterDict` or `None`
        Shared Parameters for this `Block`.
    """
    def __init__(self, batch_size, input_dim=100, nlayers=2, biflag=True,
                 dropout=0.2, input_size=800, prefix=None, params=None):
        super(BiDAFModelingLayer, self).__init__(prefix=prefix, params=params)

        self._batch_size = batch_size

        with self.name_scope():
            self._modeling_layer = LSTM(hidden_size=input_dim, num_layers=nlayers, dropout=dropout,
                                        bidirectional=biflag, input_size=input_size)

    def hybrid_forward(self, F, x, *args):  # pylint: disable=arguments-differ
        out = self._modeling_layer(x)
        return out


class BiDAFOutputLayer(HybridBlock):
    """
    ``BiDAFOutputLayer`` produces the final prediction of an answer. The output is a tuple of
    start and end index of token in the paragraph per each batch.

    It accepts 2 inputs:
        `x` : the output of Attention layer of shape:
        seq_max_length x batch_size x 8 * span_start_input_dim

        `m` : the output of Modeling layer of shape:
         seq_max_length x batch_size x 2 * span_start_input_dim

    Parameters
    ----------
    batch_size : `int`
        Size of a batch
    span_start_input_dim : `int`, default 100
        The number of features in the hidden state h of LSTM
    nlayers : `int`, default 1
        Number of recurrent layers.
    biflag: `bool`, default True
        If `True`, becomes a bidirectional RNN.
    dropout: `float`, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer.
    prefix : `str` or None
        Prefix of this `Block`.
    params : `ParameterDict` or `None`
        Shared Parameters for this `Block`.
    """
    def __init__(self, batch_size, span_start_input_dim=100, nlayers=1, biflag=True,
                 dropout=0.2, prefix=None, params=None):
        super(BiDAFOutputLayer, self).__init__(prefix=prefix, params=params)

        self._batch_size = batch_size

        with self.name_scope():
            self._dropout = nn.Dropout(rate=dropout)
            self._start_index_combined = nn.Dense(units=1, in_units=8 * span_start_input_dim,
                                                  flatten=False)
            self._start_index_model = nn.Dense(units=1, in_units=2 * span_start_input_dim,
                                               flatten=False)
            self._end_index_lstm = LSTM(hidden_size=span_start_input_dim,
                                        num_layers=nlayers, dropout=dropout, bidirectional=biflag,
                                        input_size=2 * span_start_input_dim)
            self._end_index_combined = nn.Dense(units=1, in_units=8 * span_start_input_dim,
                                                flatten=False)
            self._end_index_model = nn.Dense(units=1, in_units=2 * span_start_input_dim,
                                             flatten=False)

    def hybrid_forward(self, F, x, m, mask, *args):  # pylint: disable=arguments-differ
        # setting batch size as the first dimension
        x = F.transpose(x, axes=(1, 0, 2))

        start_index_dense_output = self._start_index_combined(self._dropout(x)) + \
                                   self._start_index_model(self._dropout(
                                       F.transpose(m, axes=(1, 0, 2))))

        m2 = self._end_index_lstm(m)
        end_index_dense_output = self._end_index_combined(self._dropout(x)) + \
                                 self._end_index_model(self._dropout(F.transpose(m2,
                                                                                 axes=(1, 0, 2))))

        start_index_dense_output = F.squeeze(start_index_dense_output)
        start_index_dense_output_masked = start_index_dense_output + ((1 - mask) *
                                                                      get_very_negative_number())

        end_index_dense_output = F.squeeze(end_index_dense_output)
        end_index_dense_output_masked = end_index_dense_output + ((1 - mask) *
                                                                  get_very_negative_number())

        return start_index_dense_output_masked, \
               end_index_dense_output_masked


class BiDAFModel(HybridBlock):
    """Bidirectional attention flow model for Question answering. Implemented according to the
    following work:

        @article{DBLP:journals/corr/abs-1804-09541,
        author    = {Minjoon Seo and
                    Aniruddha Kembhavi and
                    Ali Farhadi and
                    Hannaneh Hajishirzi},
        title     = {Bidirectional Attention Flow for Machine Comprehension},
        year      = {2016},
        url       = {https://arxiv.org/abs/1611.01603}
    }
    """
    def __init__(self, word_vocab, char_vocab, options, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._options = options

        with self.name_scope():
            self.ctx_embedding = BiDAFEmbedding(options.batch_size,
                                                word_vocab,
                                                char_vocab,
                                                options.ctx_max_len,
                                                options.ctx_embedding_num_layers,
                                                options.highway_num_layers,
                                                options.embedding_size,
                                                dropout=options.dropout,
                                                prefix='context_embedding')

            self.similarity_function = LinearSimilarity(array_1_dim=6 * options.embedding_size,
                                                        array_2_dim=1,
                                                        combination='x,y,x*y')

            self.matrix_attention = AttentionFlow(self.similarity_function,
                                                  options.batch_size,
                                                  options.ctx_max_len,
                                                  options.q_max_len,
                                                  2 * options.embedding_size)

            # we multiple embedding_size by 2 because we use bidirectional embedding
            self.attention_layer = BidirectionalAttentionFlow(options.batch_size,
                                                              options.ctx_max_len,
                                                              options.q_max_len,
                                                              2 * options.embedding_size)
            self.modeling_layer = BiDAFModelingLayer(options.batch_size,
                                                     input_dim=options.embedding_size,
                                                     nlayers=options.modeling_num_layers,
                                                     dropout=options.dropout)
            self.output_layer = BiDAFOutputLayer(options.batch_size,
                                                 span_start_input_dim=options.embedding_size,
                                                 nlayers=options.output_num_layers,
                                                 dropout=options.dropout)

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False,
                   force_reinit=False):
        super(BiDAFModel, self).initialize(init, ctx, verbose, force_reinit)
        self.ctx_embedding.init_embeddings('null' if not self._options.train_unk_token else 'write')

    def hybrid_forward(self, F, qw, cw, qc, cc, *args):  # pylint: disable=arguments-differ
        ctx_embedding_output = self.ctx_embedding(cw, cc)
        q_embedding_output = self.ctx_embedding(qw, qc)

        # attention layer expect batch_size x seq_length x channels
        ctx_embedding_output = F.transpose(ctx_embedding_output, axes=(1, 0, 2))
        q_embedding_output = F.transpose(q_embedding_output, axes=(1, 0, 2))

        # Both masks can be None
        q_mask = qw != 0
        ctx_mask = cw != 0

        passage_question_similarity = self.matrix_attention(ctx_embedding_output,
                                                            q_embedding_output)

        passage_question_similarity = passage_question_similarity.reshape(
            shape=(self._options.batch_size,
                   self._options.ctx_max_len,
                   self._options.q_max_len))

        attention_layer_output = self.attention_layer(passage_question_similarity,
                                                      ctx_embedding_output,
                                                      q_embedding_output,
                                                      q_mask,
                                                      ctx_mask)

        attention_layer_output = F.transpose(attention_layer_output, axes=(1, 0, 2))

        # modeling layer expects seq_length x batch_size x channels
        modeling_layer_output = self.modeling_layer(attention_layer_output)

        output = self.output_layer(attention_layer_output, modeling_layer_output, ctx_mask)

        return output
