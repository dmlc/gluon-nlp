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
"""Reduce LR on Plateau"""

__all__ = ['ReduceLROnPlateau']

from functools import partial

import numpy as np
from mxnet import gluon


class ReduceLROnPlateau:
    r"""Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Parameters
    ----------
    trainer : mxnet.gluon.Trainer
        Wrapped trainer.
    mode : str, default 'min'
        One of `min`, `max`. In `min` mode, lr will
        be reduced when the quantity monitored has stopped
        decreasing; in `max` mode it will be reduced when the
        quantity monitored has stopped increasing.
    factor : float, default 0.1
        Factor by which the learning rate will be
        reduced. new_lr = lr * factor.
    patience : int, default 10
        Number of epochs with no improvement after
        which learning rate will be reduced. For example, if
        `patience = 2`, then we will ignore the first 2 epochs
        with no improvement, and will only decrease the LR after the
        3rd epoch if the loss still hasn't improved then.
    verbose : bool, default False
        If True, prints a message to stdout for
        each update.
    threshold : float, default 1e-4
        Threshold for measuring the new optimum,
        to only focus on significant changes.
    threshold_mode : str, default 'rel'
        One of `rel`, `abs`. In `rel` mode,
        dynamic_threshold = best * ( 1 + threshold ) in 'max'
        mode or best * ( 1 - threshold ) in `min` mode.
        In `abs` mode, dynamic_threshold = best + threshold in
        `max` mode or best - threshold in `min` mode.
    cooldown : int, default 0
        Number of epochs to wait before resuming
        normal operation after lr has been reduced.
    min_lr : float, default 0
        A lower bound on the learning rate of all param groups
        or each group respectively.
    eps : float, default 1e-8
        Minimal decay applied to lr. If the difference
        between new and old lr is smaller than eps, the update is
        ignored.

    Examples
    --------

    >>> model = gluon.nn.Dense(10)
    >>> model.initialize()
    >>> trainer = gluon.Trainer(model.collect_params(), 'SGD')
    >>> scheduler = ReduceLROnPlateau(trainer, 'min')
    >>> for epoch in range(10): # doctest: +SKIP
    >>>     train(...) # doctest: +SKIP
    >>>     val_loss = validate(...) # doctest: +SKIP
    >>>     # Note that step should be called after validate()
    >>>     scheduler.step(val_loss) # doctest: +SKIP
    """

    def __init__(self,
                 trainer,
                 mode='min',
                 factor=0.1,
                 patience=10,
                 verbose=False,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(trainer, gluon.Trainer):
            raise TypeError('{} is not an mxnet.trainer.trainer'.format(
                type(trainer).__name__))
        self.trainer = trainer

        self.min_lr = min_lr

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode,
                             threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        r"""Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metric, epoch=None):
        r"""Function to be executed after model evaluation

        Parameters
        ----------
        metric : float
            Current metric value to mesure model performance.
        epoch : int, default None
            Current epoch. If None, it is managed internally.

        """
        current = float(metric)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        old_lr = float(self.trainer.learning_rate)
        new_lr = max(old_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.trainer.set_learning_rate(new_lr)
            if self.verbose:
                print('Epoch {:5d}: reducing learning rate'
                      ' {} to {}.'.format(epoch, old_lr, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode +
                             ' is unknown!')

        if mode == 'min':
            self.mode_worse = np.Inf
        else:  # mode == 'max':
            self.mode_worse = -np.Inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)
