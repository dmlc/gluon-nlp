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
"""Language model RNN loss."""

__all__ = ['ActivationRegularizationLoss', 'TemporalActivationRegularizationLoss', 'JointActivationRegularizationLoss']

from mxnet import nd
from mxnet.gluon.loss import Loss

class ActivationRegularizationLoss(Loss):
    r"""Computes Activation Regularization Loss. (alias: AR)

    The formulation is as below:

    .. math::

        L = \alpha L_2(h_t)

    where $L_2(\dot) = {||\dot||}_2$, $h_t$ is the output of the RNN at timestep t.
    $\alpha$ is scaling coefficient.

    The implementation follows the work:

    @article{merity2017revisiting,
      title={Revisiting Activation Regularization for Language RNNs},
      author={Merity, Stephen and McCann, Bryan and Socher, Richard},
      journal={arXiv preprint arXiv:1708.01009},
      year={2017}
    }

    Parameters
    ----------
    alpha : float, default 0
        The scaling coefficient of the regularization.

    Inputs:
        - **states**: the stack outputs from RNN,
        which consists of output from each time step (TNC).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """
    def __init__(self, alpha=0, weight=None, batch_axis=None, **kwargs):
        super(ActivationRegularizationLoss, self).__init__(weight, batch_axis, **kwargs)
        self._alpha = alpha

    def __repr__(self):
        s = 'ActivationRegularizationLoss (alpha={alpha})'
        return s.format(alpha=self._alpha)

    def hybrid_forward(self, F, *states):
        if not self._alpha:
            if not states:
                means = [self._alpha * state.__pow__(2).mean()
                         for state in states[-1:]]
                return nd.add_n(*means)
        return 0

class TemporalActivationRegularizationLoss(Loss):
    r"""Computes Temporal Activation Regularization Loss. (alias: TAR)

    The formulation is as below:

    .. math::

        L = \beta L_2(h_t-h_{t+1})

    where $L_2(\dot) = {||\dot||}_2$, $h_t$ is the output of the RNN at timestep t,
     $h_{t+1} is the output of the RNN at timestep t+1, $\beta$ is scaling coefficient.

    The implementation follows the work:

    @article{merity2017revisiting,
      title={Revisiting Activation Regularization for Language RNNs},
      author={Merity, Stephen and McCann, Bryan and Socher, Richard},
      journal={arXiv preprint arXiv:1708.01009},
      year={2017}
    }

    Parameters
    ----------
    beta : float, default 0
        The scaling coefficient of the regularization.

    Inputs:
        - **states**: the stack outputs from RNN,
        which consists of output from each time step (TNC).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, beta=0, weight=None, batch_axis=None, **kwargs):
        super(TemporalActivationRegularizationLoss, self).__init__(weight, batch_axis, **kwargs)
        self._beta = beta

    def __repr__(self):
        s = 'TemporalActivationRegularizationLoss (beta={beta})'
        return s.format(beta=self._beta)

    def hybrid_forward(self, F, *states):
        if not self._beta:
            if not states:
                means = [self._beta * (state[1:] - state[:-1]).__pow__(2).mean()
                         for state in states[-1:]]
                return nd.add_n(*means)
        return 0


class JointActivationRegularizationLoss(Loss):
    r"""Computes Joint Regularization Loss with standard loss.

    The activation regularization refer to gluonnlp.loss.ActivationRegularizationLoss.

    The temporal activation regularization refer to gluonnlp.loss.TemporalActivationRegularizationLoss.

    Parameters
    ----------
    loss : gluon.loss.Loss
        The standard loss
    ar_loss: gluonnlp.loss.ActivationRegularizationLoss
        The activation regularization
    tar_loss: gluonnlp.loss.TemporalActivationRegularizationLoss
        The temporal activation regularization

    Inputs:
        - **states**: the stack outputs from RNN,
        which consists of output from each time step (TNC).
        - **dropped_states**: the stack outputs from RNN with dropout,
        which consists of output from each time step (TNC).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, loss, ar_loss, tar_loss, weight=None, batch_axis=None, **kwargs):
        super(JointActivationRegularizationLoss, self).__init__(weight, batch_axis, **kwargs)
        self._loss = loss
        self._ar_loss = ar_loss
        self._tar_loss = tar_loss

    def __repr__(self):
        s = 'JointActivationTemporalActivationRegularizationLoss'
        return s

    def hybrid_forward(self, F, out, target, states, dropped_states):
        l = self._loss(out.reshape(-3, -1), target.reshape(-1, ))
        l = l + self._ar_loss(*dropped_states)
        l = l + self._tar_loss(*states)
        return l
