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

# pylint: disable=
r"""
This file contains some useful function and class.
"""

import json
import math


from config import opt


def mask_logits(x, mask):
    r"""Implement mask logits computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length)`
        mask : NDArray
            input tensor with shape `(batch_size, sequence_length)`

        Returns
        --------
        return : NDArray
            output tensor with shape `(batch_size, sequence_length)`
        """
    return x + -1e30 * (1 - mask)


def load_emb_mat(file_name):
    r"""Implement load embedding matrix.

        Parameters
        -----------
        file_name : string
            the embedding matrix file name.

        Returns
        --------
        mat : List[List]
            output 2-D list.
        """
    with open(opt.data_path + file_name) as f:
        mat = json.loads(f.readline())
    return mat


def warm_up_lr(step):
    r"""Implement learning rate warm up.

        Parameters
        -----------
        step : int
            control the learning rate linear increase.

        Returns
        --------
        return : int
            the learning rate for next weight update.
        """
    lr = opt.init_learning_rate
    return min(lr, lr * (math.log(step) / math.log(opt.warm_up_steps)))


def zero_grad(params):
    r"""
    Set the grad to zero.
    """
    for _, paramter in params.items():
        paramter.zero_grad()


class ExponentialMovingAverage():
    r"""An implement of Exponential Moving Average.

        shadow variable = decay * shadow variable + (1 - decay) * variable

    Parameters
    ----------
    decay : float, default 0.9999
        The axis to sum over when computing softmax and entropy.
    """

    def __init__(self, decay=0.9999, **kwargs):
        super(ExponentialMovingAverage, self).__init__(**kwargs)
        self.decay = decay
        self.shadow = {}

    def add(self, name, parameters):
        r"""Update the shadow variable.

        Parameters
        -----------
        name : string
            the name of shadow variable.
        parameters : NDArray
            the init value of shadow variable.
        Returns
        --------
        return : None
        """
        self.shadow[name] = parameters.copy()

    def __call__(self, name, x):
        r"""Update the shadow variable.

        Parameters
        -----------
        name : string
            the name of shadow variable.
        x : NDArray
            the value of shadow variable.
        Returns
        --------
        return : None
        """
        assert name in self.shadow
        self.shadow[name] = self.decay * \
            self.shadow[name] + (1.0 - self.decay) * x

    def get(self, name):
        r"""Return the shadow variable.

        Parameters
        -----------
        name : string
            the name of shadow variable.

        Returns
        --------
        return : NDArray
            the value of shadow variable.
        """
        return self.shadow[name]
