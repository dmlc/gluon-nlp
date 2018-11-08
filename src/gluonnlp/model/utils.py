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
__all__ = ['apply_weight_drop']

import collections
import functools
import re
import warnings

from mxnet.gluon import Block, contrib, rnn
from .parameter import WeightDropParameter

# pylint: disable=too-many-nested-blocks
def apply_weight_drop(block, local_param_regex, rate, axes=(),
                      weight_dropout_mode='training'):
    r"""Apply weight drop to the parameter of a block.

    Parameters
    ----------
    block : Block or HybridBlock
        The block whose parameter is to be applied weight-drop.
    local_param_regex : str
        The regex for parameter names used in the self.params.get(), such as 'weight'.
    rate : float
        Fraction of the input units to drop. Must be a number between 0 and 1.
    axes : tuple of int, default ()
        The axes on which dropout mask is shared. If empty, regular dropout is applied.
    weight_drop_mode : {'training', 'always'}, default 'training'
        Whether the weight dropout should be applied only at training time, or always be applied.

    Examples
    --------
    >>> import mxnet as mx
    >>> from mxnet import gluon
    >>> import gluonnlp as nlp
    >>> net = gluon.rnn.LSTM(10, num_layers=2, bidirectional=True)
    >>> nlp.model.apply_weight_drop(net, r'.*h2h_weight\Z', 0.5)
    >>> net.collect_params()
    lstm0_ (
      Parameter lstm0_l0_i2h_weight (shape=(40, 0), dtype=<class 'numpy.float32'>)
      WeightDropParameter lstm0_l0_h2h_weight
        (shape=(40, 10), dtype=<class 'numpy.float32'>, rate=0.5, mode=training)
      Parameter lstm0_l0_i2h_bias (shape=(40,), dtype=<class 'numpy.float32'>)
      Parameter lstm0_l0_h2h_bias (shape=(40,), dtype=<class 'numpy.float32'>)
      Parameter lstm0_r0_i2h_weight (shape=(40, 0), dtype=<class 'numpy.float32'>)
      WeightDropParameter lstm0_r0_h2h_weight
        (shape=(40, 10), dtype=<class 'numpy.float32'>, rate=0.5, mode=training)
      Parameter lstm0_r0_i2h_bias (shape=(40,), dtype=<class 'numpy.float32'>)
      Parameter lstm0_r0_h2h_bias (shape=(40,), dtype=<class 'numpy.float32'>)
      Parameter lstm0_l1_i2h_weight (shape=(40, 20), dtype=<class 'numpy.float32'>)
      WeightDropParameter lstm0_l1_h2h_weight
        (shape=(40, 10), dtype=<class 'numpy.float32'>, rate=0.5, mode=training)
      Parameter lstm0_l1_i2h_bias (shape=(40,), dtype=<class 'numpy.float32'>)
      Parameter lstm0_l1_h2h_bias (shape=(40,), dtype=<class 'numpy.float32'>)
      Parameter lstm0_r1_i2h_weight (shape=(40, 20), dtype=<class 'numpy.float32'>)
      WeightDropParameter lstm0_r1_h2h_weight
        (shape=(40, 10), dtype=<class 'numpy.float32'>, rate=0.5, mode=training)
      Parameter lstm0_r1_i2h_bias (shape=(40,), dtype=<class 'numpy.float32'>)
      Parameter lstm0_r1_h2h_bias (shape=(40,), dtype=<class 'numpy.float32'>)
    )
    >>> net.initialize()
    >>> with mx.autograd.train_mode():
    ...     print(net(mx.nd.ones((3, 4, 5))).max())
    [0.00488924]
    <NDArray 1 @cpu(0)>
    >>> with mx.autograd.train_mode():
    ...     print(net(mx.nd.ones((3, 4, 5))).max())
    [0.00475577]
    <NDArray 1 @cpu(0)>
    """
    if not rate:
        return

    existing_params = _find_params(block, local_param_regex)
    for (local_param_name, param), \
        (ref_params_list, ref_reg_params_list) in existing_params.items():
        dropped_param = WeightDropParameter(param, rate, weight_dropout_mode, axes)
        for ref_params in ref_params_list:
            ref_params[param.name] = dropped_param
        for ref_reg_params in ref_reg_params_list:
            ref_reg_params[local_param_name] = dropped_param
            if hasattr(block, local_param_name):
                local_attr = getattr(block, local_param_name)
                if local_attr == param:
                    local_attr = dropped_param
                elif isinstance(local_attr, (list, tuple)):
                    if isinstance(local_attr, tuple):
                        local_attr = list(local_attr)
                    for i, v in enumerate(local_attr):
                        if v == param:
                            local_attr[i] = dropped_param
                elif isinstance(local_attr, dict):
                    for k, v in local_attr:
                        if v == param:
                            local_attr[k] = dropped_param
                else:
                    continue
                if local_attr:
                    super(Block, block).__setattr__(local_param_name, local_attr)

# pylint: enable=too-many-nested-blocks
def _find_params(block, local_param_regex):
    # return {(local_param_name, parameter): (referenced_params_list,
    #                                         referenced_reg_params_list)}

    results = collections.defaultdict(lambda: ([], []))
    pattern = re.compile(local_param_regex)
    local_param_names = ((local_param_name, p) for local_param_name, p in block._reg_params.items()
                         if pattern.match(local_param_name))

    for local_param_name, p in local_param_names:
        ref_params_list, ref_reg_params_list = results[(local_param_name, p)]
        ref_reg_params_list.append(block._reg_params)

        params = block._params
        while params:
            if p.name in params._params:
                ref_params_list.append(params._params)
            if params._shared:
                params = params._shared
                warnings.warn('When applying weight drop, target parameter {} was found '
                              'in a shared parameter dict. The parameter attribute of the '
                              'original block on which the shared parameter dict was attached '
                              'will not be updated with WeightDropParameter. If necessary, '
                              'please update the attribute manually. The likely name of the '
                              'attribute is ".{}"'.format(p.name, local_param_name))
            else:
                break

    if block._children:
        if isinstance(block._children, list):
            children = block._children
        elif isinstance(block._children, dict):
            children = block._children.values()
        for c in children:
            child_results = _find_params(c, local_param_regex)
            for (child_p_name, child_p), (child_pd_list, child_rd_list) in child_results.items():
                pd_list, rd_list = results[(child_p_name, child_p)]
                pd_list.extend(child_pd_list)
                rd_list.extend(child_rd_list)

    return results

def _get_rnn_cell(mode, num_layers, input_size, hidden_size,
                  dropout, weight_dropout,
                  var_drop_in, var_drop_state, var_drop_out):
    """create rnn cell given specs"""
    rnn_cell = rnn.SequentialRNNCell()
    with rnn_cell.name_scope():
        for i in range(num_layers):
            if mode == 'rnn_relu':
                cell = rnn.RNNCell(hidden_size, 'relu', input_size=input_size)
            elif mode == 'rnn_tanh':
                cell = rnn.RNNCell(hidden_size, 'tanh', input_size=input_size)
            elif mode == 'lstm':
                cell = rnn.LSTMCell(hidden_size, input_size=input_size)
            elif mode == 'gru':
                cell = rnn.GRUCell(hidden_size, input_size=input_size)
            if var_drop_in + var_drop_state + var_drop_out != 0:
                cell = contrib.rnn.VariationalDropoutCell(cell,
                                                          var_drop_in,
                                                          var_drop_state,
                                                          var_drop_out)

            rnn_cell.add(cell)
            if i != num_layers - 1 and dropout != 0:
                rnn_cell.add(rnn.DropoutCell(dropout))

            if weight_dropout:
                apply_weight_drop(rnn_cell, 'h2h_weight', rate=weight_dropout)

    return rnn_cell


def _get_rnn_layer(mode, num_layers, input_size, hidden_size, dropout, weight_dropout):
    """create rnn layer given specs"""
    if mode == 'rnn_relu':
        rnn_block = functools.partial(rnn.RNN, activation='relu')
    elif mode == 'rnn_tanh':
        rnn_block = functools.partial(rnn.RNN, activation='tanh')
    elif mode == 'lstm':
        rnn_block = rnn.LSTM
    elif mode == 'gru':
        rnn_block = rnn.GRU

    block = rnn_block(hidden_size, num_layers, dropout=dropout,
                      input_size=input_size)

    if weight_dropout:
        apply_weight_drop(block, '.*h2h_weight', rate=weight_dropout)

    return block
