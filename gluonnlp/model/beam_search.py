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
"""Implements the beam search sampler."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['BeamSearchScorer', 'BeamSearchSampler']

import numpy as np
import mxnet as mx
from mxnet.gluon import HybridBlock
from .._constants import LARGE_NEGATIVE_FLOAT


class BeamSearchScorer(HybridBlock):
    r"""Score function used in beam search.

    Implements the length-penalized score function used in the GNMT paper::

        scores = (log_probs + scores) / length_penalty
        length_penalty = (K + length)^\alpha / (K + 1)^\alpha


    Parameters
    ----------
    alpha : float
    K : float
    """
    def __init__(self, alpha=1.0, K=5.0, prefix=None, params=None):
        super(BeamSearchScorer, self).__init__(prefix=prefix, params=params)
        self._alpha = alpha
        self._K = K

    def __call__(self, log_probs, scores, step):
        """Compute new scores of each candidate

        Parameters
        ----------
        log_probs : NDArray or Symbol
            The log probabilities of the candidates. Shape (d1, d2, ..., dn, V)
        scores : NDArray or Symbol
            The original scores of the beams. Shape (d1, d2, ..., dn)
        step : NDArray or Symbol
            Step to calculate the score function. It starts from 1. Shape (1,)
        Returns
        -------
        candidate_scores : NDArray or Symbol
            The scores of all the candidates. Shape (d1, d2, ..., dn, V)
        """
        return super(BeamSearchScorer, self).__call__(log_probs, scores, step)

    def hybrid_forward(self, F, log_probs, scores, step):   # pylint: disable=arguments-differ
        prev_lp = (self._K + step - 1) ** self._alpha / (self._K + 1) ** self._alpha
        prev_lp = prev_lp * (step != 1) + (step == 1)
        scores = F.broadcast_mul(scores, prev_lp)
        lp = (self._K + step) ** self._alpha / (self._K + 1) ** self._alpha
        candidate_scores = F.broadcast_add(log_probs, F.expand_dims(scores, axis=-1))
        candidate_scores = F.broadcast_div(candidate_scores, lp)
        return candidate_scores


def _expand_to_beam_size(data, beam_size, batch_size):
    """Tile all the states to have shape (beam_size, batch_size, ...)

    Parameters
    ----------
    data : A single NDArray or nested container with NDArrays
        Each NDArray should have shape (batch_size, ...)
    beam_size : int
        Beam size
    batch_size : int
        Batch size
    Returns
    -------
    new_states : Object that contains NDArrays
        Each NDArray should have shape (batch_size * beam_size, ...)
    """
    if isinstance(data, list):
        return [_expand_to_beam_size(ele, beam_size, batch_size) for ele in data]
    elif isinstance(data, tuple):
        return tuple(_expand_to_beam_size(ele, beam_size, batch_size) for ele in data)
    elif isinstance(data, dict):
        return {k: _expand_to_beam_size(v, beam_size, batch_size) for k, v in data.items()}
    elif isinstance(data, mx.nd.NDArray):
        if data.shape[0] != batch_size:
            raise ValueError('The leading dimension of all the inner elements in states must be '
                             '{}, Find shape={}'.format(batch_size, data.shape))
        return data.reshape((batch_size, 1) + data.shape[1:])\
                   .broadcast_axes(axis=1, size=beam_size)\
                   .reshape((batch_size * beam_size,) + data.shape[1:])
    else:
        raise NotImplementedError


def _choose_states(F, states, prev_states, indices):
    """

    Parameters
    ----------
    F : ndarray or symbol
    states : Object contains NDArrays/Symbols
        Each NDArray/Symbol should have shape (N, ...)
    prev_states : Object contains NDArrays/Symbols
        Each NDArray/Symbol should have shape (N, ...)
    indices : NDArray or Symbol
        Indices of the states to take. Shape (N,). If the values are in [0, N), they will be taken
        from states. If the values are in [N, 2N), they will be take from
    Returns
    -------
    new_states : Object contains NDArrays/Symbols
        Each NDArray/Symbol should have shape (N, ...)
    """
    if isinstance(states, list):
        return [_choose_states(F, lhs, rhs, indices)
                for lhs, rhs in zip(states, prev_states)]
    elif isinstance(states, tuple):
        return tuple(_choose_states(F, lhs, rhs, indices)
                     for lhs, rhs in zip(states, prev_states))
    elif isinstance(states, dict):
        return {k: _choose_states(F, states[k], prev_states[k], indices)
                for k in states}
    elif isinstance(states, (mx.nd.NDArray, mx.sym.Symbol)):
        return F.take(F.concat(states, prev_states, dim=0), indices)
    else:
        raise NotImplementedError


class _BeamSearchStepUpdate(HybridBlock):
    def __init__(self, beam_size, eos_id, scorer, prefix=None, params=None):
        super(_BeamSearchStepUpdate, self).__init__(prefix, params)
        self._beam_size = beam_size
        self._eos_id = eos_id
        self._scorer = scorer
        assert eos_id >= 0, 'eos_id cannot be negative! Received eos_id={}'.format(eos_id)

    def hybrid_forward(self, F, samples, valid_length, log_probs, scores, step, beam_alive_mask,   # pylint: disable=arguments-differ
                       states, prev_states, vocab_num, batch_size, batch_shift):
        """

        Parameters
        ----------
        F
        samples : NDArray or Symbol
            The current samples generated by beam search. Shape (batch_size, beam_size, L)
        valid_length : NDArray or Symbol
            The current valid lengths of the samples
        log_probs : NDArray or Symbol
            Log probability of the current step. Shape (batch_size * beam_size, V)
        scores : NDArray or Symbol
            The previous scores. Shape (batch_size, beam_size)
        step : NDArray or Symbol
            The current step for doing beam search. Begins from 1. Shape (1,)
        beam_alive_mask : NDArray or Symbol
            Shape (batch_size, beam_size)
        states : nested structure of NDArrays/Symbols
            Inner NDArrays have shape (batch_size * beam_size, ...)
        prev_states : nested structure of NDArrays/Symbols
            Inner NDArrays have shape (batch_size * beam_size, ...)
        vocab_num : NDArray or Symbol
            Shape (1,)
        batch_size : NDArray or Symbol
            Shape (1,)
        batch_shift : NDArray or Symbol
            Contains [0, beam_size, 2 * beam_size, (batch_size - 1) * beam_size].
            Shape (batch_size,)

        Returns
        -------
        new_samples : NDArray or Symbol
            The updated samples. Shape (batch_size, beam_size, L + 1)
        new_valid_length : NDArray or Symbol
            Valid lengths of the samples. Shape (batch_size, beam_size)
        new_scores : NDArray or Symbol
            Shape (batch_size, beam_size)
        chosen_word_ids : NDArray or Symbol
            The chosen word ids of the step. Shape (batch_size, beam_size). If it's negative,
            no word will be appended to the beam.
        beam_alive_mask : NDArray or Symbol
            Shape (batch_size, beam_size)
        new_states : nested structure of NDArrays/Symbols
            Inner NDArrays have shape (batch_size * beam_size, ...)
        """
        beam_size = self._beam_size
        beam_alive_mask_bcast = F.expand_dims(beam_alive_mask, axis=2)
        candidate_scores = self._scorer(log_probs.reshape(shape=(-4, -1, beam_size, 0)),
                                        scores, step)
        # Concat the candidate scores and the scores of the finished beams
        # The resulting candidate score will have shape (batch_size, beam_size * |V| + beam_size)
        candidate_scores = F.broadcast_mul(beam_alive_mask_bcast, candidate_scores) + \
                           F.broadcast_mul(1 - beam_alive_mask_bcast,
                                           F.ones_like(candidate_scores) * LARGE_NEGATIVE_FLOAT)
        finished_scores = F.where(beam_alive_mask,
                                  F.ones_like(scores) * LARGE_NEGATIVE_FLOAT, scores)
        candidate_scores = F.concat(candidate_scores.reshape(shape=(0, -1)),
                                    finished_scores, dim=1)
        # Get the top K scores
        new_scores, indices = F.topk(candidate_scores, axis=1, k=beam_size, ret_typ='both')
        use_prev_states = F.broadcast_greater_equal(indices, beam_size * vocab_num)
        chosen_word_ids = F.broadcast_mod(indices, vocab_num)
        beam_ids = F.where(use_prev_states,
                           F.broadcast_minus(indices, beam_size * vocab_num),
                           F.floor(F.broadcast_div(indices, vocab_num)))
        batch_beam_indices = F.broadcast_add(beam_ids, F.expand_dims(batch_shift, axis=1))
        chosen_word_ids = F.where(use_prev_states,
                                  -F.ones_like(indices),
                                  chosen_word_ids)
        # Update the samples and vaild_length
        new_samples = F.concat(F.take(samples.reshape(shape=(-3, 0)),
                                      batch_beam_indices.reshape(shape=(-1,))),
                               chosen_word_ids.reshape(shape=(-1, 1)), dim=1)\
                       .reshape(shape=(-4, -1, beam_size, 0))
        new_valid_length = F.take(valid_length.reshape(shape=(-1,)),
                                  batch_beam_indices.reshape(shape=(-1,))).reshape((-1, beam_size))\
                           + 1 - use_prev_states
        # Update the states
        new_states = _choose_states(F, states, prev_states,
                                    (batch_beam_indices +
                                     F.broadcast_mul(use_prev_states, batch_size) * beam_size)
                                    .reshape(shape=(-1,)))
        # Update the alive mask.
        beam_alive_mask = F.take(beam_alive_mask.reshape(shape=(-1,)),
                                 batch_beam_indices.reshape(shape=(-1,)))\
                              .reshape(shape=(-1, beam_size)) * (chosen_word_ids != self._eos_id)

        return new_samples, new_valid_length, new_scores,\
               chosen_word_ids, beam_alive_mask, new_states


class BeamSearchSampler(object):
    r"""Draw samples from the decoder by beam search.

    Parameters
    ----------
    beam_size : int
        The beam size.
    decoder : callable
        Function of the one-step-ahead decoder, should have the form::

            log_probs, new_states = decoder(step_input, states)

        The log_probs, input should follow these rules:

        - step_input has shape (batch_size,),
        - log_probs has shape (batch_size, V),
        - states and new_states have the same structure and the leading
          dimension of the inner NDArrays is the batch dimension.
    eos_id : int
        Id of the EOS token. No other elements will be appended to the sample if it reaches eos_id.
    scorer : BeamSearchScorer, default BeamSearchScorer(alpha=1.0, K=5)
        The score function used in beam search.
    max_length : int, default 100
        The maximum search length.
    """
    def __init__(self, beam_size, decoder, eos_id, scorer=BeamSearchScorer(alpha=1.0, K=5),
                 max_length=100):
        self._beam_size = beam_size
        assert beam_size > 0,\
            'beam_size must be larger than 0. Received beam_size={}'.format(beam_size)
        self._decoder = decoder
        self._eos_id = eos_id
        assert eos_id >= 0, 'eos_id cannot be negative! Received eos_id={}'.format(eos_id)
        self._max_length = max_length
        self._scorer = scorer
        self._updater = _BeamSearchStepUpdate(beam_size=beam_size, eos_id=eos_id, scorer=scorer)
        self._updater.hybridize()

    def __call__(self, inputs, states):
        """Sample by beam search.

        Parameters
        ----------
        inputs : NDArray
            The initial input of the decoder. Shape is (batch_size,).
        states : Object that contains NDArrays
            The initial states of the decoder.
        Returns
        -------
        samples : NDArray
            Samples draw by beam search. Shape (batch_size, beam_size, length). dtype is int32.
        scores : NDArray
            Scores of the samples. Shape (batch_size, beam_size). We make sure that scores[i, :] are
            in descending order.
        valid_length : NDArray
            The valid length of the samples. Shape (batch_size, beam_size). dtype will be int32.
        """
        batch_size = inputs.shape[0]
        beam_size = self._beam_size
        ctx = inputs.context
        batch_size_nd = mx.nd.array([batch_size], ctx=ctx)
        # Tile the states and inputs to have shape (batch_size * beam_size, ...)
        states = _expand_to_beam_size(states, beam_size=beam_size, batch_size=batch_size)
        step_input = _expand_to_beam_size(inputs, beam_size=beam_size, batch_size=batch_size)
        # All beams are initialized to alive
        # Generated samples are initialized to be the inputs
        # Except the first beam where the scores are set to be zero, all beams have -inf scores.
        # Valid length is initialized to be 1
        beam_alive_mask = mx.nd.ones(shape=(batch_size, beam_size), ctx=ctx)
        scores = mx.nd.zeros(shape=(batch_size, beam_size), ctx=ctx)
        if beam_size > 1:
            scores[:, 1:beam_size] = LARGE_NEGATIVE_FLOAT
        samples = step_input.reshape((batch_size, beam_size, 1))
        valid_length = mx.nd.ones(shape=(batch_size, beam_size), ctx=ctx)
        for i in range(self._max_length):
            log_probs, new_states = self._decoder(step_input, states)
            vocab_num_nd = mx.nd.array([log_probs.shape[1]], ctx=ctx)
            batch_shift_nd = mx.nd.arange(0, batch_size * beam_size, beam_size, ctx=ctx)
            step_nd = mx.nd.array([i + 1], ctx=ctx)
            samples, valid_length, scores, chosen_word_ids, beam_alive_mask, states = \
                self._updater(samples, valid_length, log_probs, scores, step_nd, beam_alive_mask,
                              new_states, states, vocab_num_nd, batch_size_nd, batch_shift_nd)
            step_input = mx.nd.relu(chosen_word_ids).reshape((-1,))
            if mx.nd.sum(beam_alive_mask).asscalar() == 0:
                return mx.nd.round(samples).astype(np.int32),\
                       scores,\
                       mx.nd.round(valid_length).astype(np.int32)
        final_word = mx.nd.where(beam_alive_mask,
                                 mx.nd.full(shape=(batch_size, beam_size),
                                            val=self._eos_id, ctx=ctx),
                                 mx.nd.full(shape=(batch_size, beam_size),
                                            val=-1, ctx=ctx))
        samples = mx.nd.concat(samples, final_word.reshape((0, 0, 1)), dim=2)
        valid_length += beam_alive_mask
        return mx.nd.round(samples).astype(np.int32),\
               scores,\
               mx.nd.round(valid_length).astype(np.int32)
