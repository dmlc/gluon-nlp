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

"""Building blocks and utility for models."""
__all__ = ['RNNCellLayer', 'L2Normalization']

from mxnet import ndarray
from mxnet.gluon import Block, HybridBlock


class RNNCellLayer(Block):
    """A block that takes an rnn cell and makes it act like rnn layer.

    Parameters
    ----------
    rnn_cell : Cell
        The cell to wrap into a layer-like block.
    layout : str, default 'TNC'
        The output layout of the layer.
    """
    def __init__(self, rnn_cell, layout='TNC', **kwargs):
        super(RNNCellLayer, self).__init__(**kwargs)
        self.cell = rnn_cell
        assert layout == 'TNC' or layout == 'NTC', \
            'Invalid layout %s; must be one of ["TNC" or "NTC"]'%layout
        self._layout = layout
        self._axis = layout.find('T')
        self._batch_axis = layout.find('N')

    def forward(self, inputs, states=None): # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        batch_size = inputs.shape[self._batch_axis]
        skip_states = states is None
        if skip_states:
            states = self.cell.begin_state(batch_size, ctx=inputs.context)
        if isinstance(states, ndarray.NDArray):
            states = [states]
        for state, info in zip(states, self.cell.state_info(batch_size)):
            if state.shape != info['shape']:
                raise ValueError(
                    'Invalid recurrent state shape. Expecting %s, got %s.'%(
                        str(info['shape']), str(state.shape)))
        states = sum(zip(*((j for j in i) for i in states)), ())
        outputs, states = self.cell.unroll(
            inputs.shape[self._axis], inputs, states,
            layout=self._layout, merge_outputs=True)

        if skip_states:
            return outputs
        return outputs, states


class L2Normalization(HybridBlock):
    """Normalize the input array by dividing the L2 norm along the given axis.

    ..code

        out = data / (sqrt(sum(data**2, axis)) + eps)

    Parameters
    ----------
    axis : int, default -1
        The axis to compute the norm value.
    eps : float, default 1E-6
        The epsilon value to avoid dividing zero
    """
    def __init__(self, axis=-1, eps=1E-6, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self._axis = axis
        self._eps = eps

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        ret = F.broadcast_div(x, F.norm(x, axis=self._axis, keepdims=True) + self._eps)
        return ret

class SampledLogits(Block):
    """Block that passes through the input directly.

    This block can be used in conjunction with HybridConcurrent
    block for residual connection.

            This under-estimates the full softmax and is only used for training.

    Example::

        inputs = ...
        # for testing
        test = nn.Dense(inputs, ...)
        # for training
        train = contrib.nn.SampledLogits(inputs, params=test.params)

    Parameters
    ----------
    num_classes: int
        Number of possible classes.
    num_sampled: int
        Number of classes randomly sampled for each batch.
    in_unit: int
        Dimensionality of the input space.
    remove_accidental_hits: bool
        Whether to remove "accidental hits" when a sampled candidate is equal to
        one of the true classes.
    """
    def __init__(self, num_classes, num_sampled, in_unit, remove_accidental_hits,
                 prefix=None, params=None):
        super(SampledLogits, self).__init__(prefix=prefix, params=params)
        self.weight = self.params.get('weight', shape=(num_classes, in_unit),
                                      #init=weight_initializer, #dtype=dtype,
                                      grad_stype='row_sparse', stype='row_sparse')

        self.bias = self.params.get('bias', shape=(num_classes, 1),
                                    grad_stype='row_sparse', stype='row_sparse')
        # TODO make them private
        self._num_classes = num_classes
        self._num_sampled = num_sampled
        self._in_unit = in_unit
        self._remove_accidental_hits = remove_accidental_hits

    def forward(self, x, sampled_candidates, expected_count_sampled,
                expected_count_true, label):
        F = ndarray
        # (n,)
        label = F.reshape(label, shape=(-1,))
        import numpy as np
        sampled_candidates = sampled_candidates.astype(np.float32)
        # (num_sampled+n, )
        ids_all = F.concat(sampled_candidates, label, dim=0)
        # lookup weights and biases
        weight = self.weight.row_sparse_data(ids_all)
        bias = self.bias.row_sparse_data(ids_all)
        # (num_sampled+n, dim)
        w_all = F.Embedding(data=ids_all, weight=weight,
                            input_dim=self._num_classes, output_dim=self._in_unit,
                            sparse_grad=True)
        # (num_sampled+n, 1)
        b_all = F.Embedding(data=ids_all, weight=bias,
                            input_dim=self._num_classes, output_dim=1,
                            sparse_grad=True)
        # (num_sampled, dim)
        w_sampled = w_all.slice(begin=(0, 0), end=(self._num_sampled, None))
        w_true = w_all.slice(begin=(self._num_sampled, 0), end=(None, None))
        b_sampled = b_all.slice(begin=(0, 0), end=(self._num_sampled, None))
        b_true = b_all.slice(begin=(self._num_sampled, 0), end=(None, None))
        # true
        # (n, 1)
        x = x.reshape((-1, self._in_unit))
        logits_true = (w_true * x).sum(axis=1, keepdims=True) + b_true
        # samples
        # (n, num_sampled)
        b_sampled = F.reshape(b_sampled, (-1,))
        logits_sampled = F.FullyConnected(x, weight=w_sampled, bias=b_sampled,
                                          num_hidden=self._num_sampled)

        # remove accidental hits
        if self._remove_accidental_hits:
            label_vec = F.reshape(label, (-1, 1))
            sample_vec = F.reshape(sampled_candidates, (1, -1))
            mask = F.broadcast_equal(label_vec, sample_vec) * -1e37
            logits_sampled = logits_sampled + mask

        expected_count_sampled = F.reshape(expected_count_sampled,
                                           shape=(1, self._num_sampled))
        expected_count_true = expected_count_true.reshape((-1, 1))
        expected_count_true = F.cast(expected_count_true, np.float32)
        logits_true = logits_true - F.log(expected_count_true)
        expected_count_sampled = F.cast(expected_count_sampled, np.float32)
        logits_sampled = F.broadcast_sub(logits_sampled, F.log(expected_count_sampled))

        # return logits and new_labels
        # (n, 1+num_sampled)
        logits = F.concat(logits_true, logits_sampled, dim=1)
        return logits, F.zeros_like(label)

    def __repr__(self):
        s = '{name}({mapping})'
        mapping = '{0} -> {1}, with {2} samples'.format(self._in_unit, self._num_classes,
                                                        self._num_sampled)
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **self.__dict__)
