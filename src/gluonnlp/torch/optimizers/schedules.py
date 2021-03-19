# Copyright 2020, Amazon.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimization for BERT model."""

from torch.optim.lr_scheduler import LambdaLR

__all__ = ['get_warmup_linear_const_decay_poly_schedule']


def get_warmup_linear_const_decay_poly_schedule(optimizer, total_steps, warmup_ratio=0.002,
                                                const_ratio=0., degree=1.0, last_epoch=-1):
    """Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period during which it
    increases linearly from 0 to the initial lr set in the optimizer and a
    constant period.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        total_steps (:obj:`int`):
            The total number of training steps.
        warmup_ratio (:obj:`float`):
            The number of steps for the warmup phase.
        constant_ratio (:obj:`float`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """
    def lr_lambda(global_step: int):
        x = global_step / total_steps
        if warmup_ratio == 0.0:
            return 1.0
        elif x < warmup_ratio:
            return x / warmup_ratio
        elif x < warmup_ratio + const_ratio:
            return 1.0
        return ((1.0 - x) / (1.0 - warmup_ratio - const_ratio))**degree

    return LambdaLR(optimizer, lr_lambda, last_epoch)
