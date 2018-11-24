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

"""Bidirectional attention flow layer"""
from mxnet import gluon
import numpy as np

from .utils import last_dim_softmax, replace_masked_values, masked_softmax


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
