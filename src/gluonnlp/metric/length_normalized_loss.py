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
""" Length Normalized Loss """

from mxnet import ndarray
from mxnet.metric import EvalMetric

__all__ = ['LengthNormalizedLoss']

class LengthNormalizedLoss(EvalMetric):
    """Compute length normalized loss metrics

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, axis=0, name='length-normalized-loss',
                 output_names=None, label_names=None):
        super(LengthNormalizedLoss, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names,
            has_global_stats=True)

    def update(self, labels, preds):
        """Update the length normalized metrics with target label and loss

        Update the sum_metrics and sum_insts of metrics with provided arguments.

        Parameters:
        ----------
        labels: list or tuple
            It contains two elements. The first element is the target sentence and
            the second element is the valid length of target sentence
        preds: list or ndarray.ndarray.NDArray
            a list of ndarray.ndarray.NDArray or scalar or a single
            ndarray.ndarray.NDArray. It is usually the loss predicted by the model
        """
        typecheck = not isinstance(labels, list) and not isinstance(labels, tuple)
        if typecheck or len(labels) != 2:
            raise ValueError('labels must be a list. Its first element should be'
                             ' target sequence and the second element should be'
                             'the valid length of sequence.')

        _, seq_valid_length = labels

        if not isinstance(seq_valid_length, list):
            seq_valid_length = [seq_valid_length]

        if not isinstance(preds, list):
            preds = [preds]

        for length in seq_valid_length:
            if isinstance(length, ndarray.ndarray.NDArray):
                total_length = ndarray.sum(length).asscalar()
            else:
                total_length = length
            self.num_inst += total_length
            self.global_num_inst += total_length

        for pred in preds:
            if isinstance(pred, ndarray.ndarray.NDArray):
                loss = ndarray.sum(pred).asscalar()
            else:
                loss = pred
            self.sum_metric += loss
            self.global_sum_metric += loss
