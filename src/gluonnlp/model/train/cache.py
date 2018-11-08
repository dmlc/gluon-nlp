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
"""Cache model."""
__all__ = ['CacheCell']

import mxnet as mx

from mxnet import nd
from mxnet.gluon import Block

class CacheCell(Block):
    r"""Cache language model.

    We implement the neural cache language model proposed in the following work::

        @article{grave2016improving,
        title={Improving neural language models with a continuous cache},
        author={Grave, Edouard and Joulin, Armand and Usunier, Nicolas},
        journal={ICLR},
        year={2017}
        }

    Parameters
    ----------
    lm_model : gluonnlp.model.StandardRNN or gluonnlp.model.AWDRNN
        The type of RNN to use. Options are 'gluonnlp.model.StandardRNN', 'gluonnlp.model.AWDRNN'.
    vocab_size : int
        Size of the input vocabulary.
    window : int
        Size of cache window
    theta : float
        The scala controls the flatness of the cache distribution
        that predict the next word as shown below:

        .. math::

            p_{cache} \propto \sum_{i=1}^{t-1} \mathbb{1}_{w=x_{i+1}} exp(\theta {h_t}^T h_i)

        where :math:`p_{cache}` is the cache distribution, :math:`\mathbb{1}` is
        the identity function, and :math:`h_i` is the output of timestep i.
    lambdas : float
        Linear scalar between only cache and vocab distribution, the formulation is as below:

        .. math::

            p = (1 - \lambda) p_{vocab} + \lambda p_{cache}

        where :math:`p_{vocab}` is the vocabulary distribution and :math:`p_{cache}`
        is the cache distribution.
    """
    def __init__(self, lm_model, vocab_size, window, theta, lambdas, **kwargs):
        super(CacheCell, self).__init__(**kwargs)
        self._vocab_size = vocab_size
        self._window = window
        self._theta = theta
        self._lambdas = lambdas
        with self.name_scope():
            self.lm_model = lm_model

    def save_parameters(self, filename):
        """Save parameters to file.

        filename : str
            Path to file.
        """
        self.lm_model.save_parameters(filename)

    def load_parameters(self, filename, ctx=mx.cpu()): # pylint: disable=arguments-differ
        """Load parameters from file.

        filename : str
            Path to parameter file.
        ctx : Context or list of Context, default cpu()
            Context(s) initialize loaded parameters on.
        """
        self.lm_model.load_parameters(filename, ctx=ctx)

    def begin_state(self, *args, **kwargs):
        """Initialize the hidden states.
        """
        return self.lm_model.begin_state(*args, **kwargs)


    def forward(self, inputs, target, next_word_history, cache_history, begin_state=None): # pylint: disable=arguments-differ
        """Defines the forward computation for cache cell. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`.

        Parameters
        ----------
        inputs: NDArray
            The input data
        target: NDArray
            The label
        next_word_history: NDArray
            The next word in memory
        cache_history: NDArray
            The hidden state in cache history


        Returns
        --------
        out: NDArray
            The linear interpolation of the cache language model
            with the regular word-level language model
        next_word_history: NDArray
            The next words to be kept in the memory for look up
            (size is equal to the window size)
        cache_history: NDArray
            The hidden states to be kept in the memory for look up
            (size is equal to the window size)
        """
        output, hidden, encoder_hs, _ = \
            super(self.lm_model.__class__, self.lm_model).\
                forward(inputs, begin_state)
        encoder_h = encoder_hs[-1].reshape(-3, -2)
        output = output.reshape(-1, self._vocab_size)

        start_idx = len(next_word_history) \
            if next_word_history is not None else 0
        next_word_history = nd.concat(*[nd.one_hot(t[0], self._vocab_size, on_value=1, off_value=0)
                                        for t in target], dim=0) if next_word_history is None \
            else nd.concat(next_word_history,
                           nd.concat(*[nd.one_hot(t[0], self._vocab_size, on_value=1, off_value=0)
                                       for t in target], dim=0), dim=0)
        cache_history = encoder_h if cache_history is None \
            else nd.concat(cache_history, encoder_h, dim=0)

        out = None
        softmax_output = nd.softmax(output)
        for idx, vocab_L in enumerate(softmax_output):
            joint_p = vocab_L
            if start_idx + idx > self._window:
                valid_next_word = next_word_history[start_idx + idx - self._window:start_idx + idx]
                valid_cache_history = cache_history[start_idx + idx - self._window:start_idx + idx]
                logits = nd.dot(valid_cache_history, encoder_h[idx])
                cache_attn = nd.softmax(self._theta * logits).reshape(-1, 1)
                cache_dist = (cache_attn.broadcast_to(valid_next_word.shape)
                              * valid_next_word).sum(axis=0)
                joint_p = self._lambdas * cache_dist + (1 - self._lambdas) * vocab_L

            out = joint_p[target[idx]] if out is None \
                else nd.concat(out, joint_p[target[idx]], dim=0)
        next_word_history = next_word_history[-self._window:]
        cache_history = cache_history[-self._window:]
        return out, next_word_history, cache_history, hidden
