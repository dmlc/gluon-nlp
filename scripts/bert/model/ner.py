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
"""Gluon model block for the named entity recognition task."""

import mxnet as mx
from mxnet.gluon import Block, nn


class BERTTagger(Block):
    """Model for sequence tagging with BERT

    Parameters
    ----------
    bert_model: BERTModel
        Bidirectional encoder with transformer.
    num_tag_types: int
        number of possible tags
    dropout_prob: float
        dropout probability for the last layer
    prefix: str or None
        See document of `mx.gluon.Block`.
    params: ParameterDict or None
        See document of `mx.gluon.Block`.
    """

    def __init__(self, bert_model, num_tag_types, dropout_prob, prefix=None, params=None):
        super(BERTTagger, self).__init__(prefix=prefix, params=params)
        self.bert_model = bert_model
        with self.name_scope():
            self.tag_classifier = nn.Dense(units=num_tag_types, flatten=False)
            self.dropout = nn.Dropout(rate=dropout_prob)

    def forward(self, token_ids, token_types, valid_length): # pylint: disable=arguments-differ
        """Generate an unnormalized score for the tag of each token

        Parameters
        ----------
        token_ids: NDArray, shape (batch_size, seq_length)
            ID of tokens in sentences
            See `input` of `glounnlp.model.BERTModel`
        token_types: NDArray, shape (batch_size, seq_length)
            See `glounnlp.model.BERTModel`
        valid_length: NDArray, shape (batch_size,)
            See `glounnlp.model.BERTModel`

        Returns
        -------
        NDArray, shape (batch_size, seq_length, num_tag_types):
            Unnormalized prediction scores for each tag on each position.
        """
        bert_output = self.dropout(self.bert_model(token_ids, token_types, valid_length))
        output = self.tag_classifier(bert_output)
        return output


def attach_prediction(data_loader, net, ctx, is_train):
    """Attach the prediction from a model to a data loader as the last field.

    Parameters
    ----------
    data_loader: mx.gluon.data.DataLoader
        Input data from `bert_model.BERTTaggingDataset._encode_as_input`.
    net: mx.gluon.Block
        gluon `Block` for making the preciction.
    ctx:
        The context data should be loaded to.
    is_train:
        Whether the forward pass should be made with `mx.autograd.record()`.

    Returns
    -------
        All fields from `bert_model.BERTTaggingDataset._encode_as_input`,
        as well as the prediction of the model.

    """
    for data in data_loader:
        text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag = \
            [x.astype('float32').as_in_context(ctx) for x in data]

        from contextlib import ExitStack
        with ExitStack() as stack:
            if is_train:
                stack.enter_context(mx.autograd.record())
            out = net(text_ids, token_types, valid_length)
        yield text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag, out
