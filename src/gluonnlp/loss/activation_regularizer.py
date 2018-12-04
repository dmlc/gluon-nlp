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

__all__ = ['ActivationRegularizationLoss', 'TemporalActivationRegularizationLoss']

from mxnet.gluon.loss import Loss


class ActivationRegularizationLoss(Loss):
    r"""Computes Activation Regularization Loss. (alias: AR)

    The formulation is as below:

    .. math::

        L = \alpha L_2(h_t)

    where :math:`L_2(\cdot) = {||\cdot||}_2, h_t` is the output of the RNN at timestep t.
    :math:`\alpha` is scaling coefficient.

    The implementation follows the work::

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
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, alpha=0, weight=None, batch_axis=None, **kwargs):
        super(ActivationRegularizationLoss, self).__init__(weight, batch_axis, **kwargs)
        self._alpha = alpha

    def __repr__(self):
        s = 'ActivationRegularizationLoss (alpha={alpha})'
        return s.format(alpha=self._alpha)

    def hybrid_forward(self, F, *states): # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        states : list
            the stack outputs from RNN, which consists of output from each time step (TNC).

        Returns
        --------
        loss : NDArray
            loss tensor with shape (batch_size,). Dimensions other than batch_axis are averaged out.
        """
        # pylint: disable=unused-argument
        if self._alpha != 0:
            if states:
                means = [self._alpha * state.__pow__(2).mean()
                         for state in states[-1:]]
                return F.add_n(*means)
            else:
                return F.zeros(1)
        return F.zeros(1)


class TemporalActivationRegularizationLoss(Loss):
    r"""Computes Temporal Activation Regularization Loss. (alias: TAR)

    The formulation is as below:

    .. math::

        L = \beta L_2(h_t-h_{t+1})

    where :math:`L_2(\cdot) = {||\cdot||}_2, h_t` is the output of the RNN at timestep t,
    :math:`h_{t+1}` is the output of the RNN at timestep t+1, :math:`\beta` is scaling coefficient.

    The implementation follows the work::

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
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """

    def __init__(self, beta=0, weight=None, batch_axis=None, **kwargs):
        super(TemporalActivationRegularizationLoss, self).__init__(weight, batch_axis, **kwargs)
        self._beta = beta

    def __repr__(self):
        s = 'TemporalActivationRegularizationLoss (beta={beta})'
        return s.format(beta=self._beta)

    def hybrid_forward(self, F, *states): # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        states : list
            the stack outputs from RNN, which consists of output from each time step (TNC).

        Returns
        --------
        loss : NDArray
            loss tensor with shape (batch_size,). Dimensions other than batch_axis are averaged out.
        """
        # pylint: disable=unused-argument
        if self._beta != 0:
            if states:
                means = [self._beta * (state[1:] - state[:-1]).__pow__(2).mean()
                         for state in states[-1:]]
                return F.add_n(*means)
            else:
                return F.zeros(1)
        return F.zeros(1)
