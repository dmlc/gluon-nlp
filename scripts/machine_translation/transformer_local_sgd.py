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

# coding: utf-8
# pylint: disable=line-too-long
"""Parameter optimizer."""

from mxnet import optimizer as opt
from mxnet.model import _create_kvstore, _create_sparse_kvstore
from mxnet.gluon.parameter import ParameterDict, Parameter

import mxnet as mx
import types
import warnings
import math

class LocalSGDTrainer(mx.gluon.Trainer):
    """Local Adam optimizer for Transformer.

    Parameters
    ----------
    local_sgd_interval : int, default 1
        If local_sgd_interval<=1, run fully synchronous SGD,
        otherwise, sync params and states for every local_sgd_interval steps.
    local_sgd_regularization : float, default 0
        The weight of local regularization, within the range [0, 1)
    local_sgd_regularization_interval : int, default 0
        If larger than 0, add the regularization term to the local solver 
        after every local_sgd_regularization_interval steps
    """
    def __init__(self, params, optimizer, optimizer_params=None, kvstore='device', 
                local_sgd_interval=1, local_sgd_regularization=0, local_sgd_regularization_interval=0):

        super(LocalSGDTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=kvstore, update_on_kvstore=False)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by Horovod size, which is equivalent to performing
        # average in allreduce, has better performance. 
        if local_sgd_interval is None or local_sgd_interval <= 1:
            self._local_sgd_interval = 1
        else:
            self._local_sgd_interval = local_sgd_interval
        self._local_sgd_counter = 0
        update_on_kvstore = False
        self._local_sgd_regularization = local_sgd_regularization
        self._local_sgd_regularization_interval = local_sgd_regularization_interval
        self._local_sgd_regularization_counter = 0
        self._is_states_initialized = False

    def _init_kvstore(self):
        """Create kvstore."""
        config = self._kvstore_params
        # configure kvstore, update_on_kvstore and self._distributed on three cases:
        if self._contains_sparse_weight:
            # If weight is sparse, kvstore must be present and the weight must be updated on kvstore.
            # The training loop is the following:
            #    - row_sparse_pull(sparse_weight)
            #    - forward()
            #    - backward()
            #    - push_and_update(grad)
            #    - pull(weight)
            kvstore, update_on_kvstore = _create_sparse_kvstore(config['kvstore'])
            self._distributed = 'dist' in kvstore.type
            # raise err if user provides unsupported configs
            if config['update_on_kvstore'] is False:
                raise ValueError("Cannot set update_on_kvstore=False when sparse weights "
                                 "are present.")

        elif self._contains_sparse_grad:
            # For single node training with dense weight and sparse grad,
            # we prefer update_on_kvstore=False because this is usually faster.
            # This means we push and pull sparse gradients, and we do not store weight in kvstore.
            # The training loop is the following:
            #    - forward()
            #    - backward()
            #    - push(grad)
            #    - pull(grad)
            #    - update(grad, weight)
            #
            # For multi-node training with dense weight and sparse grad,
            # only update_on_kvstore=True is supported, due to the fact that
            # kv.row_sparse_pull(grad) is not implemented.
            # Therefore, we push sparse gradients and pull dense weights.
            # The training loop contains:
            #    - forward()
            #    - backward()
            #    - push_and_update(grad)
            #    - pull(weight)
            arg_arrays = {param.name: param.data(self._contexts[0]) for param in self._params}
            kvstore, _ = _create_kvstore(config['kvstore'], len(self._contexts), arg_arrays)
            self._distributed = 'dist' in kvstore.type if kvstore else False
            update_on_kvstore = self._distributed
            # raise err if user provides unsupported configs
            if config['update_on_kvstore'] is not None:
                if config['update_on_kvstore'] is False and self._distributed:
                    raise ValueError("Cannot set update_on_kvstore=False on dist kvstore "
                                     "when sparse gradients are present.")
                update_on_kvstore = config['update_on_kvstore']

        else:
            # Training with dense weight and dense gradients.
            # The only unsupported mode is async with update_on_kvstore=False
            arg_arrays = {param.name: param.data(self._contexts[0]) for param in self._params}
            if self._local_sgd_interval > 1:
                # local sgd
                state_arrays = {param.name+'_state': param.data(self._contexts[0]) for param in self._params}
                arg_arrays.update(state_arrays)
            kvstore, update_on_kvstore = _create_kvstore(config['kvstore'], len(self._contexts),
                                                         arg_arrays)
            self._distributed = 'dist' in kvstore.type if kvstore else False
            if self._distributed and 'async' in kvstore.type:
                update_on_kvstore = True
                # raise err if user provides unsupported configs
                if config['update_on_kvstore'] is False:
                    raise ValueError("Please set update_on_kvstore=True "
                                     "when training in async mode.")
            if config['update_on_kvstore'] is not None:
                update_on_kvstore = config['update_on_kvstore']

        # set grad compression and optimizers
        if kvstore:
            if self._compression_params:
                kvstore.set_gradient_compression(self._compression_params)
            if update_on_kvstore:
                # optimizer preferably needs to be set before init for multiprecision
                kvstore.set_optimizer(self._optimizer)
            self._kvstore = kvstore
            self._update_on_kvstore = update_on_kvstore
        else:
            self._kvstore = None
            self._update_on_kvstore = None

        self._kv_initialized = True

    def reset_adam_counter(self, t):
        print(self._updaters[1].optimizer._index_update_count)
        print(t)
        # for i, param in enumerate(self._params):
        #     if param.grad_req != 'null':
        #         for updater in self._updaters:
        #             updater.optimizer._index_update_count[i] = t

    def init_states(self):
        """Initialize states (momentum for sgd_mon, or mean/var for adam) in the KVStore, for local sgd
        """
        assert self._kv_initialized, "Cannot initialize states in KVStore " \
                                     "when KVStore is not initialized."
        if self._kvstore and self._is_states_initialized == False:
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    if isinstance(self._updaters[0].states[i], (tuple, list)):
                        # for some optimizers, there are multiple states (mean, variance), such as Adam
                        # TODO(xcong) there might be some other side cases
                        for j in range(len(self._updaters[0].states[i])):
                            state_arrays = [updater.states[i][j] for updater in self._updaters]
                            self._kvstore.init(i+len(self._params)*(j+1), self._updaters[0].states[i][j])
                    else:
                        state_arrays = [updater.states[i] for updater in self._updaters]
                        self._kvstore.init(i+len(self._params), self._updaters[0].states[i])
            self._is_states_initialized = True

    def step(self, batch_sizes, ignore_stale_grad=False):
        """Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.

        Parameters
        ----------
        batch_sizes : [int]
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        # rescale_grad = self._scale / batch_size
        # self._check_and_rescale_grad(rescale_grad)

        # rescale the grads
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                for j in range(len(batch_sizes)):
                    param.list_grad()[j] /= batch_sizes[j]

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        if self._local_sgd_interval == 1:
            # if not local sgd
            self._allreduce_grads()

        if self._local_sgd_interval > 1 and self._local_sgd_counter == 0 and self._local_sgd_regularization > 0:
            # regularization for local sgd
            self._local_sgd_regularization_params = []
            for i, param in enumerate(self._params):
                if param.grad_req != 'null' and param._stype == 'default':
                    self._local_sgd_regularization_params.append([self._local_sgd_regularization * x.copy() for x in param.list_data()])
                else:
                    self._local_sgd_regularization_params.append([])

        self._update(ignore_stale_grad)

        if self._local_sgd_interval > 1 and self._local_sgd_regularization > 0:
            # regularization for local sgd
            # TODO(xcong): use param.name instead of the indices
            mixing_weight = (1 - self._local_sgd_regularization)
            self._local_sgd_regularization_counter += 1
            if self._local_sgd_regularization_interval == 0 or self._local_sgd_regularization_interval == self._local_sgd_regularization_counter:
                self._local_sgd_regularization_counter = 0
                for i, param in enumerate(self._params):
                    if param.grad_req != 'null' and param._stype == 'default':
                        for j, data in enumerate(param.list_data()):
                            data *= mixing_weight
                            data += self._local_sgd_regularization_params[i][j]

        if self._local_sgd_interval > 1:
            # local sgd
            self._local_sgd_counter += 1
            if self._local_sgd_counter == self._local_sgd_interval:
                self._local_sgd_counter = 0
                # synchronization
                self._allreduce_params()
                if self._is_states_initialized:
                    self._allreduce_states()
                # indicate that the parameters are synchronized in the current iteration
                return True
            return False
        return True

    def allreduce_params(self):
        """For each parameter, reduce the gradients from different contexts.

        Should be called after `autograd.backward()`, outside of `record()` scope,
        and before `trainer.update()`.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        """
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
        
        self._allreduce_params()

    def _allreduce_params(self):
        # print("_allreduce_params")
        if self._kvstore:
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    self._kvstore.push(i, param.list_data(), priority=-i)
                    if param._stype == 'default':
                        self._kvstore.pull(i, param.list_data(), priority=-i)
                        # take average
                        # assume that every worker has the same number of gpus/contexts
                        num_workers = self._kvstore.num_workers * len(param.list_data())
                        for data in param.list_data():
                            data /= num_workers
                    else:
                        raise ValueError("Cannot pull row_sparse parameters for local SGD")

    def allreduce_states(self):
        """For each parameter, reduce the gradients from different contexts.

        Should be called after `autograd.backward()`, outside of `record()` scope,
        and before `trainer.update()`.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        """
        if not self._kv_initialized:
            self._init_kvstore()
        
        if not self._is_states_initialized:
            raise ValueError("States are not initiallized")
        self._allreduce_states()

    def _allreduce_states(self):
        # print("_allreduce_states")
        if self._kvstore:
            # for i, param in enumerate(self._params):
            for i, param in reversed(list(enumerate(self._params))):
                if param.grad_req != 'null':
                    if isinstance(self._updaters[0].states[i], (tuple, list)):
                        # for some optimizers, there are multiple states (mean, variance), such as Adam
                        for j in range(len(self._updaters[0].states[i])):
                            state_arrays = [updater.states[i][j] for updater in self._updaters]
                            idx = i+len(self._params)*(j+1)
                            self._kvstore.push(idx, state_arrays, priority=i-len(self._params)*2)
                            if param._stype == 'default':
                                self._kvstore.pull(idx, state_arrays, priority=i-len(self._params)*2)
                                # take average
                                # assume that every worker has the same number of gpus/contexts
                                num_workers = float(self._kvstore.num_workers * len(state_arrays))
                                for state in state_arrays:
                                    state /= num_workers

                            else:
                                raise ValueError("Cannot pull row_sparse parameters for local SGD")
                    else:
                        state_arrays = [updater.states[i] for updater in self._updaters]
                        idx = i+len(self._params)
                        self._kvstore.push(idx, state_arrays, priority=i-len(self._params)*2)
                        if param._stype == 'default':
                            self._kvstore.pull(idx, state_arrays, priority=i-len(self._params)*2)
                            # take average
                            # assume that every worker has the same number of gpus/contexts
                            num_workers = self._kvstore.num_workers * len(state_arrays)
                            for state in state_arrays:
                                state /= num_workers
                        else:
                            raise ValueError("Cannot pull row_sparse parameters for local SGD")

