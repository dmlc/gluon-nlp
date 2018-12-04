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
__all__ = ['WeightDropParameter']

from mxnet import nd, gluon


class WeightDropParameter(gluon.Parameter):
    """A Container holding parameters (weights) of Blocks and performs dropout.

    Parameters
    ----------
    parameter : Parameter
        The parameter which drops out.
    rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
        Dropout is not applied if dropout_rate is 0.
    mode : str, default 'training'
        Whether to only turn on dropout during training or to also turn on for inference.
        Options are 'training' and 'always'.
    axes : tuple of int, default ()
        Axes on which dropout mask is shared.
    """
    def __init__(self, parameter, rate=0.0, mode='training', axes=()):
        p = parameter
        super(WeightDropParameter, self).__init__(
            name=p.name, grad_req=p.grad_req, shape=p._shape, dtype=p.dtype,
            lr_mult=p.lr_mult, wd_mult=p.wd_mult, init=p.init,
            allow_deferred_init=p._allow_deferred_init,
            differentiable=p._differentiable)
        self._rate = rate
        self._mode = mode
        self._axes = axes

    def data(self, ctx=None):
        """Returns a copy of this parameter on one context. Must have been
        initialized on this context before.

        Parameters
        ----------
        ctx : Context
            Desired context.
        Returns
        -------
        NDArray on ctx
        """
        d = self._check_and_get(self._data, ctx)
        if self._rate:
            d = nd.Dropout(d, self._rate, self._mode, self._axes)
        return d

    def __repr__(self):
        s = 'WeightDropParameter {name} (shape={shape}, dtype={dtype}, rate={rate}, mode={mode})'
        return s.format(name=self.name, shape=self.shape, dtype=self.dtype,
                        rate=self._rate, mode=self._mode)
