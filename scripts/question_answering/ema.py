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

"""Exponential moving average"""
import mxnet as mx

from mxnet import gluon


class ExponentialMovingAverage(object):
    r"""An implement of Exponential Moving Average.

        shadow variable = decay * shadow variable + (1 - decay) * variable

    Parameters
    ----------
    decay : float, default 0.9999
        The axis to sum over when computing softmax and entropy.
    """

    def __init__(self, decay=0.9999, **kwargs):
        super(ExponentialMovingAverage, self).__init__(**kwargs)
        self._params = None
        self._decay = decay
        self._shadow_params = gluon.ParameterDict()

    def initialize(self, params):
        """Initialize EMA. Usually it should be called after 1st forward pass happened as
        EMA requires shape information to be available.

        Parameters
        ----------
        params : `ParameterDict`
            Parameters of the network, usually obtained by calling net.collect_params()
        """
        self._params = params

        for param in self._params.values():
            shadow_param = self._shadow_params.get(param.name, shape=param.shape)
            shadow_param.initialize(mx.init.Constant(self._param_data_to_cpu(param)), ctx=mx.cpu())

    def update(self):
        """
        Updates currently held saved parameters with current state of network.
        All calculations for this average occur on the cpu context.
        """
        for param in self._params.values():
            shadow_param = self._shadow_params.get(param.name)
            shadow_param.set_data(
                (1 - self._decay) * self._param_data_to_cpu(param) +
                self._decay * shadow_param.data(mx.cpu()))

    def get_param(self, name):
        """Return the shadow variable.

        Parameters
        -----------
        name : string
            the name of shadow variable.

        Returns
        --------
        return : NDArray
            the value of shadow variable.
        """
        return self._shadow_params.get(name).data(mx.cpu())

    def get_params(self):
        """ Provides averaged parameters

        Returns
        -------
        gluon.ParameterDict
            Averaged parameters
        """
        return self._shadow_params

    def _param_data_to_cpu(self, param):
        """Returns a copy (on CPU context) of the data held in some context of given parameter.

        Parameters
        ----------
        param: gluon.Parameter
            Parameter's whose data needs to be copied.

        Returns
        -------
        NDArray
            Copy of data on CPU context.
        """
        return param.list_data()[0].copyto(mx.cpu())
