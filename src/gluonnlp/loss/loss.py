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

"""Loss functions."""

__all__ = ['MaskedSoftmaxCrossEntropyLoss', 'MaskedSoftmaxCELoss']

import numpy as np
from mxnet.gluon.loss import SoftmaxCELoss

class MaskedSoftmaxCrossEntropyLoss(SoftmaxCELoss):
    r"""Wrapper of the SoftmaxCELoss that supports valid_length as the input
    (alias: MaskedSoftmaxCELoss)

    If `sparse_label` is `True` (default), label should contain integer
    category indicators:

    .. math::

        \DeclareMathOperator{softmax}{softmax}

        p = \softmax({pred})

        L = -\sum_i \log p_{i,{label}_i}

    `label`'s shape should be `pred`'s shape with the channel dimension removed.
    i.e. for `pred` with shape (1,2,3) `label`'s shape should
    be (1,2).

    If `sparse_label` is `False`, `label` should contain probability distribution
    and `label`'s shape should be the same with `pred`:

    .. math::

        p = \softmax({pred})

        L = -\sum_i \sum_j {label}_j \log p_{ij}

    Parameters
    ----------
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.

    Inputs:
        - **pred**: the prediction tensor, shape should be (N, T, C)
        - **label**: the truth tensor. When `sparse_label` is True, `label`'s
          shape should be `pred`'s shape with the channel dimension C removed.
          i.e. for `pred` with shape (1,2,3) `label`'s shape should be (1,2)
          and values should be integers between 0 and 2.
          If `sparse_label` is False, `label`'s shape must be the same as `pred`
          and values should be floats in the range `[0, 1]`.
        - **valid_length**: valid length of each sequence, of shape (batch_size, )
          predictions elements longer than their valid_length are masked out

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """
    def __init__(self, sparse_label=True, from_logits=False, weight=None,
                 **kwargs):
        # The current technique only works with NTC data
        axis = -1
        batch_axis = 0
        super(MaskedSoftmaxCrossEntropyLoss, self).__init__(axis, sparse_label, from_logits,
                                                            weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, valid_length): # pylint: disable=arguments-differ
        if self._sparse_label:
            sample_weight = F.cast(F.expand_dims(F.ones_like(label), axis=-1), dtype=np.float32)
        else:
            sample_weight = F.ones_like(label)
        sample_weight = F.SequenceMask(sample_weight,
                                       sequence_length=valid_length,
                                       use_sequence_length=True,
                                       axis=1)
        return super(MaskedSoftmaxCrossEntropyLoss, self).hybrid_forward(
            F, pred, label, sample_weight)

MaskedSoftmaxCELoss = MaskedSoftmaxCrossEntropyLoss
