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
"""Bidirectional attention flow model with all subblocks"""
import numpy as np
from mxnet import gluon
from mxnet import initializer
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn

from gluonnlp.model import ConvolutionalEncoder, Highway, BiLMEncoder

try:
    from similarity_function import DotProductSimilarity, LinearSimilarity
except ImportError:
    from .similarity_function import DotProductSimilarity, LinearSimilarity


class AttentionFlow(gluon.HybridBlock):
    """
    This ``block`` takes two ndarrays as input and returns a ndarray of attentions.

    We compute the similarity between each row in each matrix and return unnormalized similarity
    scores.  Because these scores are unnormalized, we don't take a mask as input; it's up to the
    caller to deal with masking properly when this output is used.

    By default similarity is computed with a dot product, but you can alternatively use a
    parameterized similarity function if you wish.


    Input:
        - ndarray_1: ``(batch_size, num_rows_1, embedding_dim)``
        - ndarray_2: ``(batch_size, num_rows_2, embedding_dim)``

    Output:
        - ``(batch_size, num_rows_1, num_rows_2)``

    Parameters
    ----------
    similarity_function: ``SimilarityFunction``, optional (default=``DotProductSimilarity``)
        The similarity function to use when computing the attention.
    """

    def __init__(self, similarity_function, passage_length,
                 question_length, **kwargs):
        super(AttentionFlow, self).__init__(**kwargs)

        self._similarity_function = similarity_function or DotProductSimilarity()
        self._passage_length = passage_length
        self._question_length = question_length

    def hybrid_forward(self, F, matrix_1, matrix_2):
        # pylint: disable=arguments-differ,unused-argument,missing-docstring
        tiled_matrix_1 = F.broadcast_axis(matrix_1.expand_dims(2), axis=2,
                                          size=self._question_length)

        tiled_matrix_2 = F.broadcast_axis(matrix_2.expand_dims(1), axis=1,
                                          size=self._passage_length)

        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)


class BidirectionalAttentionFlow(gluon.HybridBlock):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).
    """

    def __init__(self,
                 passage_length,
                 question_length,
                 **kwargs):
        super(BidirectionalAttentionFlow, self).__init__(**kwargs)

        self._passage_length = passage_length
        self._question_length = question_length

    def _get_big_negative_value(self):
        """Provides maximum negative Float32 value
        Returns
        -------
        value : float32
            Maximum negative float32 value
        """
        return np.finfo(np.float32).min

    def _get_small_positive_value(self):
        """Provides minimal possible Float32 value
        Returns
        -------
        value : float32
            Minimal float32 value
        """
        return np.finfo(np.float32).eps

    def hybrid_forward(self, F, passage_question_similarity,
                       encoded_passage, encoded_question, question_mask, passage_mask):
        # pylint: disable=arguments-differ,missing-docstring
        # Shape: (batch_size, passage_length, question_length)
        question_mask_reshaped = F.broadcast_axis(question_mask.expand_dims(1), axis=1,
                                                  size=self._passage_length)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = last_dim_softmax(F,
                                                      passage_question_similarity,
                                                      question_mask_reshaped,
                                                      epsilon=self._get_small_positive_value())
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = F.batch_dot(passage_question_attention, encoded_question)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = passage_question_similarity if question_mask is None else \
            replace_masked_values(F,
                                  passage_question_similarity,
                                  question_mask.expand_dims(1),
                                  replace_with=self._get_big_negative_value())

        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(axis=-1)

        # Shape: (batch_size, passage_length)
        question_passage_attention = masked_softmax(F, question_passage_similarity, passage_mask,
                                                    epsilon=self._get_small_positive_value())

        # Shape: (batch_size, encoding_dim)
        question_passage_vector = F.squeeze(F.batch_dot(question_passage_attention.expand_dims(1),
                                                        encoded_passage), axis=1)

        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.expand_dims(1)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = F.concat(encoded_passage,
                                        passage_question_vectors,
                                        encoded_passage * passage_question_vectors,
                                        F.broadcast_mul(encoded_passage,
                                                        tiled_question_passage_vector),
                                        dim=-1)

        return final_merged_passage


class BiDAFEmbedding(HybridBlock):
    """BiDAFEmbedding is a class describing embeddings that are separately applied to question
    and context of the datasource. Both question and context are passed in two NDArrays:
    1. Matrix of words: batch_size x words_per_question/context
    2. Tensor of characters: batch_size x words_per_question/context x chars_per_word
    """

    def __init__(self, word_vocab, char_vocab, max_seq_len,
                 contextual_embedding_nlayers=2, highway_nlayers=2, embedding_size=100,
                 dropout=0.2, prefix=None, params=None):
        super(BiDAFEmbedding, self).__init__(prefix=prefix, params=params)

        self._word_vocab = word_vocab
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
            self._contextual_embedding = BiLMEncoder(mode='lstm',
                                                     num_layers=contextual_embedding_nlayers,
                                                     input_size=2 * embedding_size,
                                                     hidden_size=embedding_size,
                                                     dropout=dropout,
                                                     skip_connection=False)

    def init_embeddings(self, grad_req='null'):
        """Initialize words embeddings with provided embedding values

        Parameters
        ----------
        grad_req: str
            How to treat gradients of embedding layer
        """
        self._word_embedding.weight.set_data(self._word_vocab.embedding.idx_to_vec)
        self._word_embedding.collect_params().setattr('grad_req', grad_req)

    def hybrid_forward(self, F, w, c, ctx_embedding_state, word_mask):
        # pylint: disable=arguments-differ,missing-docstring
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

        # batch_size x seq_len x embedding
        highway_input = F.concat(char_embedded, word_embedded, dim=2)

        def highway(token_of_all_batches, _):
            return self._highway_network(token_of_all_batches), []

        # Pass through highway, shape remains unchanged
        highway_output, _ = F.contrib.foreach(highway, highway_input, [])

        ce_output, _ = self._contextual_embedding(highway_output, ctx_embedding_state, word_mask)
        ce_output = ce_output.slice_axis(axis=0, begin=-1, end=None).squeeze(axis=0)
        return ce_output


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

    def __init__(self, span_start_input_dim=100, nlayers=1,
                 dropout=0.2, prefix=None, params=None):
        super(BiDAFOutputLayer, self).__init__(prefix=prefix, params=params)

        with self.name_scope():
            self._dropout = nn.Dropout(rate=dropout)
            self._start_index_combined = nn.Dense(units=1, in_units=8 * span_start_input_dim,
                                                  flatten=False)
            self._start_index_model = nn.Dense(units=1, in_units=2 * span_start_input_dim,
                                               flatten=False)
            self._end_index_lstm = BiLMEncoder(mode='lstm',
                                               num_layers=nlayers,
                                               input_size=2 * span_start_input_dim,
                                               hidden_size=span_start_input_dim,
                                               dropout=dropout,
                                               skip_connection=False)
            self._end_index_combined = nn.Dense(units=1, in_units=8 * span_start_input_dim,
                                                flatten=False)
            self._end_index_model = nn.Dense(units=1, in_units=2 * span_start_input_dim,
                                             flatten=False)

    def hybrid_forward(self, F, x, m, end_index_states, mask):
        # pylint: disable=arguments-differ,missing-docstring
        # setting batch size as the first dimension
        x = F.transpose(x, axes=(1, 0, 2))

        start_index_dense_output = self._start_index_combined(self._dropout(x)) + \
                                   self._start_index_model(self._dropout(
                                       F.transpose(m, axes=(1, 0, 2))))

        m2, _ = self._end_index_lstm(m, end_index_states, mask)
        m2 = m2.slice_axis(axis=0, begin=-1, end=None).squeeze(axis=0)
        end_index_dense_output = self._end_index_combined(self._dropout(x)) + \
                                 self._end_index_model(self._dropout(F.transpose(m2,
                                                                                 axes=(1, 0, 2))))

        start_index_dense_output = F.squeeze(start_index_dense_output)
        start_index_dense_output_masked = start_index_dense_output + ((1 - mask) * -1e30)

        end_index_dense_output = F.squeeze(end_index_dense_output)
        end_index_dense_output_masked = end_index_dense_output + ((1 - mask) * -1e30)

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
        super(BiDAFModel, self).__init__(prefix=prefix, params=params)
        self._options = options
        self._padding_token_idx = word_vocab[word_vocab.padding_token]

        with self.name_scope():
            self.ctx_embedding = BiDAFEmbedding(word_vocab,
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
                                                  options.ctx_max_len,
                                                  options.q_max_len)

            # we multiple embedding_size by 2 because we use bidirectional embedding
            self.attention_layer = BidirectionalAttentionFlow(options.ctx_max_len,
                                                              options.q_max_len)

            self.modeling_layer = BiLMEncoder(mode='lstm',
                                              num_layers=1,
                                              input_size=8 * options.embedding_size,
                                              hidden_size=options.embedding_size,
                                              dropout=0.0,
                                              skip_connection=False)

            self.output_layer = BiDAFOutputLayer(span_start_input_dim=options.embedding_size,
                                                 nlayers=options.output_num_layers,
                                                 dropout=options.dropout)

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False,
                   force_reinit=False):
        super(BiDAFModel, self).initialize(init, ctx, verbose, force_reinit)
        self.ctx_embedding.init_embeddings('null' if not self._options.train_unk_token else 'write')

    def hybrid_forward(self, F, qw, cw, qc, cc,
                       ctx_embedding_state, modeling_layer_state, end_index_state):
        # pylint: disable=arguments-differ
        # Both masks can be None
        q_mask = qw != self._padding_token_idx
        ctx_mask = cw != self._padding_token_idx

        # pylint: disable=arguments-differ,missing-docstring
        ctx_embedding_output = self.ctx_embedding(cw, cc, ctx_embedding_state, ctx_mask)
        q_embedding_output = self.ctx_embedding(qw, qc, ctx_embedding_state, q_mask)

        # attention layer expect batch_size x seq_length x channels
        ctx_embedding_output = F.transpose(ctx_embedding_output, axes=(1, 0, 2))
        q_embedding_output = F.transpose(q_embedding_output, axes=(1, 0, 2))

        passage_question_similarity = self.matrix_attention(ctx_embedding_output,
                                                            q_embedding_output)

        passage_question_similarity = passage_question_similarity.reshape(
            shape=(0,
                   self._options.ctx_max_len,
                   self._options.q_max_len))

        attention_layer_output = self.attention_layer(passage_question_similarity,
                                                      ctx_embedding_output,
                                                      q_embedding_output,
                                                      q_mask,
                                                      ctx_mask)

        attention_layer_output = F.transpose(attention_layer_output, axes=(1, 0, 2))

        # modeling layer expects seq_length x batch_size x channels
        modeling_layer_output, _ = self.modeling_layer(attention_layer_output,
                                                       modeling_layer_state,
                                                       ctx_mask)

        ml_last_layer = modeling_layer_output.slice_axis(axis=0, begin=-1, end=None).squeeze(axis=0)

        output = self.output_layer(attention_layer_output,
                                   ml_last_layer,
                                   end_index_state,
                                   ctx_mask)

        return output


def last_dimension_applicator(F,
                              function_to_apply,
                              tensor,
                              mask,
                              **kwargs):
    """
    Takes a tensor with 3 or more dimensions and applies a function over the last dimension.  We
    assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask
    is of same shape . We flatten both tensor and mask to be 2D, pass them through
    the function and put the tensor back in its original shape.
    """
    reshaped_tensor = tensor.reshape(shape=(-3, -1))

    if mask is not None:
        mask = mask.reshape(shape=(-3, -1))

    reshaped_result = function_to_apply(F, reshaped_tensor, mask, **kwargs)
    return F.reshape_like(lhs=reshaped_result, rhs=tensor)


def masked_softmax(F, vector, mask, epsilon):
    """
    ``nd.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is None:
        result = F.softmax(vector, axis=-1)
    else:
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = F.softmax(vector * mask, axis=-1)
        result = result * mask
        result = F.broadcast_div(result, (result.sum(axis=1, keepdims=True) + epsilon))
    return result


def last_dim_softmax(F, tensor, mask, epsilon):
    """
    Takes a tensor with 3 or more dimensions and does a masked softmax over the last dimension.  We
    assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask (if given)
    has shape ``(batch_size, sequence_length)``.
    """
    return last_dimension_applicator(F, masked_softmax, tensor, mask, epsilon=epsilon)


def replace_masked_values(F, tensor, mask, replace_with):
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.
    """
    # We'll build a tensor of the same shape as `tensor`, zero out masked values, then add back in
    # the `replace_with` value.
    one_minus_mask = 1.0 - mask
    values_to_add = replace_with * one_minus_mask
    return F.broadcast_add(F.broadcast_mul(tensor, mask), values_to_add)
