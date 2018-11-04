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
from mxnet.lr_scheduler import LRScheduler


class ExponentialScheduler(LRScheduler):
    def __init__(self, base_lr=0.01, decay_rate=0.5, decay_every=1, warmup_steps=0, warmup_begin_lr=0,
                 warmup_mode='linear'):
        """
        A simple learning rate decay scheduler
        :param base_lr: the initial learning rate.
        :param decay_rate: how much does the learning rate decreases to in every decay
        :param decay_every: how often does the decay occurs
        :param warmup_steps: not used
        :param warmup_begin_lr: not used
        :param warmup_mode: not used
        """
        super().__init__(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.decay_rate = decay_rate
        self.decay_every = decay_every

    def __call__(self, num_update):
        return self.base_lr * self.decay_rate ** (num_update / self.decay_every)
