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
from mxnet.gluon import Block, HybridBlock, contrib
from mxnet.gluon.contrib.nn import SparseEmbedding


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

import mxnet as mx
import math
import numpy as np

class ZipfianSampler(mx.operator.CustomOp):
    def __init__(self, num_sampled, range_max):
        self.num_sampled = num_sampled
        self.range_max = range_max

    def prob_helper(self, num_tries, num_sampled, prob):
        if num_tries == num_sampled:
            return prob * num_sampled
        return (num_tries * (-prob).log1p()).expm1() * -1

    def forward(self, is_train, req, in_data, out_data, aux):
        range_max = self.range_max
        num_sampled = self.num_sampled
        true_classes = in_data[0]
        ctx = true_classes.context
        # implementation
        log_range = math.log(range_max + 1)
        num_tries = 0
        output_set = set()
        true_classes = true_classes.reshape((-1,))
        sampled_classes = []

        while len(sampled_classes) < num_sampled:
            rand = np.random.uniform(low=0, high=log_range)
            c = np.asscalar((np.exp(rand) - 1).astype('int64') % range_max)
            num_tries += 1
            if c not in output_set:
                output_set.add(c)
                sampled_classes.append(c)

        true_cls = true_classes.as_in_context(ctx).astype('float64')
        prob_true = ((true_cls + 2.0) / (true_cls + 1.0)).log() / log_range
        count_true = self.prob_helper(num_tries, num_sampled, prob_true)

        sampled_classes = ndarray.array(sampled_classes, ctx=ctx, dtype='int64')
        sampled_cls_fp64 = sampled_classes.astype('float64')
        prob_sampled = ((sampled_cls_fp64 + 2.0) / (sampled_cls_fp64 + 1.0)).log() / log_range
        count_sampled = self.prob_helper(num_tries, num_sampled, prob_sampled)

        self.assign(out_data[0], req[0], sampled_classes)
        self.assign(out_data[1], req[1], count_true)
        self.assign(out_data[2], req[2], count_sampled)

@mx.operator.register("rand_zipfian_sampler")
class ZipfianProp(mx.operator.CustomOpProp):
    def __init__(self, num_sampled, range_max):
        super(ZipfianProp, self).__init__(need_top_grad=False)
        self.num_sampled = int(num_sampled)
        self.range_max = int(range_max)

    def list_arguments(self):
        return ['true_classes']

    def list_outputs(self):
        return ['sampled_classes', 'expected_count_true', 'expected_count_sampled']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (self.num_sampled, )
        return [data_shape], [output_shape, data_shape, output_shape], []

    def infer_type(self, in_type):
        return in_type, [np.int64, np.float64, np.float64], []

    def create_operator(self, ctx, shapes, dtypes):
        return ZipfianSampler(self.num_sampled, self.range_max)
