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
"""DRMM models."""

from mxnet.gluon import nn
import numpy as np


class DRMM(nn.HybridBlock):
    """A Deep Relevance Matching Model for Ad-hoc Retrieval

    https://arxiv.org/abs/1711.08611

    We implement the DRMM model proposed in the following work::

        @article{guo2016a,
            title={A Deep Relevance Matching Model for Ad-hoc Retrieval},
            author={Guo, Jiafeng and Fan, Yixing and Ai, Qingyao and Croft, W Bruce},
            journal={conference on information and knowledge management},
            pages={55--64},
            year={2016}
        }

    Parameters
    ----------
    vocab_size: int
        Number of words in vocab.
    embed_size: int
        Dimension of word vector.
    num_layers: int, default 2
        Number of dense layer.
    hidden_sizes: Tuple[int], default [10,1]
        The hidden units of each dense layer, the last one should be 1.
    hist_size: int, default 30
        The number of bins of the matching histogram.
    output_size: int, default 2
        Number of categories.
    pad_val: int, default 1
        Pad value in the input, used to calculate Mask.
    hist_type: str, default 'LCH'
        The type of the historgram, it should be one of 'NH' or 'LCH'.

    """

    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_layers=2,
                 hidden_sizes=(10, 1),
                 hist_size=30,
                 output_size=2,
                 pad_val=1,
                 hist_type='LCH',
                 **kwargs):
        super(DRMM, self).__init__(**kwargs)
        self.hist_size = hist_size
        self.pad_val = pad_val
        if hist_type in ['LCH', 'NH']:
            self.hist_type = hist_type
        else:
            raise ValueError(
                'hist_type \'' + hist_type +
                '\' not understood. only \'LCH\',\'NH\' are supported.')
        with self.name_scope():
            self.ffw = nn.HybridSequential()
            with self.ffw.name_scope():
                for i in range(num_layers):
                    self.ffw.add(
                        nn.Dense(hidden_sizes[i],
                                 activation='tanh',
                                 flatten=False))
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.attentions = nn.Dense(1, use_bias=False, flatten=False)
            self.output = nn.Dense(output_size)

    def hist_map(self, F, embed_query, embed_doc, mask):
        """
        Parameters
        ----------
        embed_query : NDArray
            Shape (batch_size, query_length, embe_size)
        embed_doc : NDArray
            Shape (batch_size, doc_length, embe_size)
        mask: NdArray
            Shape (batch_size, query_length, doc_length)

        Returns
        -------
        hist : NDArray
            Shape (batch_size, query_length, hist_size)
        """

        mm = F.batch_dot(embed_query, F.transpose(embed_doc, (0, 2, 1)))
        norm1 = F.expand_dims(F.norm(embed_query, axis=2), axis=2)
        norm2 = F.expand_dims(F.norm(embed_doc, axis=2), axis=2)
        n_n = F.batch_dot(norm1, F.transpose(norm2, (0, 2, 1)))
        cosine_distance = mm / (n_n + mask)

        bin_upperbounds = np.linspace(-1, 1, num=self.hist_size)[1:]
        H = []

        for bin_upperbound in bin_upperbounds:
            H.append((cosine_distance < bin_upperbound).sum(axis=-1) + 1)
        H.append(((cosine_distance > 0.999) *
                  (cosine_distance < 1.001)).sum(axis=-1) + 1)
        matching_hist = F.stack(*H, axis=2)

        if self.hist_type == 'NH':
            matching_hist_sum = matching_hist.sum(axis=-1)
            hist = F.broadcast_div(matching_hist,
                                   F.expand_dims(matching_hist_sum, axis=2))

        if self.hist_type == 'LCH':
            hist = F.log(matching_hist)

        return hist

    def hybrid_forward(self, F, query, doc):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        query : NDArray
            Shape (batch_size, query_length)
        doc : NDArray
            Shape (batch_size, doc_length)

        Returns
        -------
        out : NDArray
            Shape (batch_size, output_size)
        """
        embed_query = self.embedding(query)
        embed_doc = self.embedding(doc)

        # shape(batch_size, length, embed_size)
        query_mask = F.where(query != self.pad_val, F.zeros_like(query),
                             float('-inf') *
                             F.ones_like(query)).astype('float32')
        doc_mask = F.where(doc != self.pad_val, F.zeros_like(doc),
                           float('-inf') * F.ones_like(doc)).astype('float32')

        # shape(batch_size, query_length, doc_length)
        _mask = F.broadcast_add(F.expand_dims(doc_mask, axis=1),
                                F.expand_dims(query_mask, axis=2))
        _mask = F.where(_mask == 0, F.zeros_like(_mask),
                        float('-inf') * F.ones_like(_mask)).astype('float32')

        # shape(batch_size, query_length, hist_size)
        hist = self.hist_map(F, embed_query, embed_doc, _mask)

        # shape(batch_size, query_length, hist_size) -> shape(batch, query_length)
        x = self.ffw(hist).squeeze()

        # shape(batch_size, query_length, embed_size) -> shape(batch_size, query_length)
        w = self.attentions(embed_query).squeeze() + query_mask
        w = F.softmax(w)

        out = self.output(w * x)
        return out
