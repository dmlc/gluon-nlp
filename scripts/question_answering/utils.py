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

"""Various utility methods for Question Answering"""
import math


def warm_up_lr(base_lr, iteration, lr_warmup_steps):
    """Returns learning rate based on current iteration.

    This function is used to implement learning rate warm up technique.

    math::

      lr = min(base_lr, base_lr * (log(iteration) /  log(lr_warmup_steps)))

    Parameters
    ----------
    base_lr : float
        Initial learning rage
    iteration : int
        Current iteration number
    lr_warmup_steps : int
        Learning rate warm up steps

    Returns
    -------
    learning_rate : float
        Learning rate
    """
    return min(base_lr, base_lr * (math.log(iteration) / math.log(lr_warmup_steps)))
