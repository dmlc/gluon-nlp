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

"""Attention Flow Layer"""
from mxnet import gluon

from .similarity_function import DotProductSimilarity


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
