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

__all__ = ['BeamSearchScorer', 'BeamSearchSampler', 'HybridBeamSearchSampler']

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
    alpha : float, default 1.0
    K : float, default 5.0
    """
    def __init__(self, alpha=1.0, K=5.0, prefix=None, params=None):
        super(BeamSearchScorer, self).__init__(prefix=prefix, params=params)
        self._alpha = alpha
        self._K = K

    def __call__(self, log_probs, scores, step): # pylint: disable=arguments-differ
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


def _extract_and_flatten_nested_structure(data, flattened=None):
    """Flatten the structure of a nested container to a list.

    Parameters
    ----------
    data : A single NDArray/Symbol or nested container with NDArrays/Symbol.
        The nested container to be flattened.
    flattened : list or None
        The container thats holds flattened result.
    Returns
    -------
    structure : An integer or a nested container with integers.
        The extracted structure of the container of `data`.
    flattened : (optional) list
        The container thats holds flattened result.
        It is returned only when the input argument `flattened` is not given.
    """
    if flattened is None:
        flattened = []
        structure = _extract_and_flatten_nested_structure(data, flattened)
        return structure, flattened
    if isinstance(data, list):
        return list(_extract_and_flatten_nested_structure(x, flattened) for x in data)
    elif isinstance(data, tuple):
        return tuple(_extract_and_flatten_nested_structure(x, flattened) for x in data)
    elif isinstance(data, dict):
        return {k: _extract_and_flatten_nested_structure(v) for k, v in data.items()}
    elif isinstance(data, (mx.sym.Symbol, mx.nd.NDArray)):
        flattened.append(data)
        return len(flattened) - 1
    else:
        raise NotImplementedError


def _reconstruct_flattened_structure(structure, flattened):
    """Reconstruct the flattened list back to (possibly) nested structure.

    Parameters
    ----------
    structure : An integer or a nested container with integers.
        The extracted structure of the container of `data`.
    flattened : list or None
        The container thats holds flattened result.
    Returns
    -------
    data : A single NDArray/Symbol or nested container with NDArrays/Symbol.
        The nested container that was flattened.
    """
    if isinstance(structure, list):
        return list(_reconstruct_flattened_structure(x, flattened) for x in structure)
    elif isinstance(structure, tuple):
        return tuple(_reconstruct_flattened_structure(x, flattened) for x in structure)
    elif isinstance(structure, dict):
        return {k: _reconstruct_flattened_structure(v, flattened) for k, v in structure.items()}
    elif isinstance(structure, int):
        return flattened[structure]
    else:
        raise NotImplementedError


def _expand_to_beam_size(data, beam_size, batch_size, state_info=None):
    """Tile all the states to have batch_size * beam_size on the batch axis.

    Parameters
    ----------
    data : A single NDArray/Symbol or nested container with NDArrays/Symbol
        Each NDArray/Symbol should have shape (N, ...) when state_info is None,
        or same as the layout in state_info when it's not None.
    beam_size : int
        Beam size
    batch_size : int
        Batch size
    state_info : Nested structure of dictionary, default None.
        Descriptors for states, usually from decoder's ``state_info()``.
        When None, this method assumes that the batch axis is the first dimension.
    Returns
    -------
    new_states : Object that contains NDArrays/Symbols
        Each NDArray/Symbol should have shape batch_size * beam_size on the batch axis.
    """
    assert not state_info or isinstance(state_info, (type(data), dict)), \
            'data and state_info doesn\'t match, ' \
            'got: {} vs {}.'.format(type(state_info), type(data))
    if isinstance(data, list):
        if not state_info:
            state_info = [None] * len(data)
        return [_expand_to_beam_size(d, beam_size, batch_size, s)
                for d, s in zip(data, state_info)]
    elif isinstance(data, tuple):
        if not state_info:
            state_info = [None] * len(data)
            state_info = tuple(state_info)
        return tuple(_expand_to_beam_size(d, beam_size, batch_size, s)
                     for d, s in zip(data, state_info))
    elif isinstance(data, dict):
        if not state_info:
            state_info = {k: None for k in data.keys()}
        return {k: _expand_to_beam_size(v, beam_size, batch_size, state_info[k])
                for k, v in data.items()}
    elif isinstance(data, mx.nd.NDArray):
        if not state_info:
            batch_axis = 0
        else:
            batch_axis = state_info['__layout__'].find('N')
        if data.shape[batch_axis] != batch_size:
            raise ValueError('The batch dimension of all the inner elements in states must be '
                             '{}, Found shape={}'.format(batch_size, data.shape))
        new_shape = list(data.shape)
        new_shape[batch_axis] = batch_size * beam_size
        new_shape = tuple(new_shape)
        return data.expand_dims(batch_axis+1)\
                   .broadcast_axes(axis=batch_axis+1, size=beam_size)\
                   .reshape(new_shape)
    elif isinstance(data, mx.sym.Symbol):
        if not state_info:
            batch_axis = 0
        else:
            batch_axis = state_info['__layout__'].find('N')
        new_shape = (0, ) * batch_axis + (-3, -2)
        return data.expand_dims(batch_axis+1)\
                   .broadcast_axes(axis=batch_axis+1, size=beam_size)\
                   .reshape(new_shape)
    else:
        raise NotImplementedError


def _choose_states(F, states, state_info, indices):
    """

    Parameters
    ----------
    F : ndarray or symbol
    states : Object contains NDArrays/Symbols
        Each NDArray/Symbol should have shape (N, ...) when state_info is None,
        or same as the layout in state_info when it's not None.
    state_info : Nested structure of dictionary, default None.
        Descriptors for states, usually from decoder's ``state_info()``.
        When None, this method assumes that the batch axis is the first dimension.
    indices : NDArray or Symbol
        Indices of the states to take. Shape (N,).
    Returns
    -------
    new_states : Object contains NDArrays/Symbols
        Each NDArray/Symbol should have shape (N, ...).
    """
    assert not state_info or isinstance(state_info, (type(states), dict)), \
            'states and state_info don\'t match'
    if isinstance(states, list):
        if not state_info:
            state_info = [None] * len(states)
        return [_choose_states(F, d, s, indices) for d, s in zip(states, state_info)]
    elif isinstance(states, tuple):
        if not state_info:
            state_info = [None] * len(states)
            state_info = tuple(state_info)
        return tuple(_choose_states(F, d, s, indices) for d, s in zip(states, state_info))
    elif isinstance(states, dict):
        if not state_info:
            state_info = {k: None for k in states.keys()}
        return {k: _choose_states(F, v, state_info[k], indices)
                for k, v in states.items()}
    elif isinstance(states, (mx.nd.NDArray, mx.sym.Symbol)):
        if not state_info:
            batch_axis = 0
        else:
            batch_axis = state_info['__layout__'].find('N')
        if batch_axis != 0:
            states = states.swapaxes(0, batch_axis)
        states = F.take(states, indices)
        if batch_axis != 0:
            states = states.swapaxes(0, batch_axis)
        return states
    else:
        raise NotImplementedError


class _BeamSearchStepUpdate(HybridBlock):
    def __init__(self, beam_size, eos_id, scorer, state_info, single_step=False, \
        prefix=None, params=None):
        super(_BeamSearchStepUpdate, self).__init__(prefix, params)
        self._beam_size = beam_size
        self._eos_id = eos_id
        self._scorer = scorer
        self._state_info = state_info
        self._single_step = single_step
        assert eos_id >= 0, 'eos_id cannot be negative! Received eos_id={}'.format(eos_id)

    def hybrid_forward(self, F, samples, valid_length, log_probs, scores, step, beam_alive_mask,  # pylint: disable=arguments-differ
                       states, vocab_size, batch_shift):
        """

        Parameters
        ----------
        F
        samples : NDArray or Symbol
            The current samples generated by beam search.
            When single_step is True, (batch_size, beam_size, max_length).
            When single_step is False, (batch_size, beam_size, L).
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
            Each NDArray/Symbol should have shape (N, ...) when state_info is None,
            or same as the layout in state_info when it's not None.
        vocab_size : NDArray or Symbol
            Shape (1,)
        batch_shift : NDArray or Symbol
            Contains [0, beam_size, 2 * beam_size, ..., (batch_size - 1) * beam_size].
            Shape (batch_size,)

        Returns
        -------
        new_samples : NDArray or Symbol or an empty list
            The updated samples.
            When single_step is True, it is an empty list.
            When single_step is False, shape (batch_size, beam_size, L + 1)
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
        use_prev = F.broadcast_greater_equal(indices, beam_size * vocab_size)
        chosen_word_ids = F.broadcast_mod(indices, vocab_size)
        beam_ids = F.where(use_prev,
                           F.broadcast_minus(indices, beam_size * vocab_size),
                           F.floor(F.broadcast_div(indices, vocab_size)))
        batch_beam_indices = F.broadcast_add(beam_ids, F.expand_dims(batch_shift, axis=1))
        chosen_word_ids = F.where(use_prev,
                                  -F.ones_like(indices),
                                  chosen_word_ids)
        # Update the samples and vaild_length
        new_samples = F.concat(F.take(samples.reshape(shape=(-3, 0)),
                                      batch_beam_indices.reshape(shape=(-1,))),
                               chosen_word_ids.reshape(shape=(-1, 1)), dim=1)\
                       .reshape(shape=(-4, -1, beam_size, 0))
        if self._single_step:
            new_samples = new_samples.slice_axis(axis=2, begin=1, end=None)
        new_valid_length = F.take(valid_length.reshape(shape=(-1,)),
                                  batch_beam_indices.reshape(shape=(-1,))).reshape((-1, beam_size))\
                           + 1 - use_prev
        # Update the states
        new_states = _choose_states(F, states, self._state_info, batch_beam_indices.reshape((-1,)))
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
        if hasattr(decoder, 'state_info'):
            state_info = decoder.state_info()
        else:
            state_info = None
        self._updater = _BeamSearchStepUpdate(beam_size=beam_size, eos_id=eos_id, scorer=scorer,
                                              state_info=state_info)
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
        # Tile the states and inputs to have shape (batch_size * beam_size, ...)
        if hasattr(self._decoder, 'state_info'):
            state_info = self._decoder.state_info(batch_size)
        else:
            state_info = None
        states = _expand_to_beam_size(states, beam_size=beam_size, batch_size=batch_size,
                                      state_info=state_info)
        step_input = _expand_to_beam_size(inputs, beam_size=beam_size, batch_size=batch_size)
        # All beams are initialized to alive
        # Generated samples are initialized to be the inputs
        # Except the first beam where the scores are set to be zero, all beams have -inf scores.
        # Valid length is initialized to be 1
        beam_alive_mask = mx.nd.ones(shape=(batch_size, beam_size), ctx=ctx)
        valid_length = mx.nd.ones(shape=(batch_size, beam_size), ctx=ctx)
        scores = mx.nd.zeros(shape=(batch_size, beam_size), ctx=ctx)
        if beam_size > 1:
            scores[:, 1:beam_size] = LARGE_NEGATIVE_FLOAT
        samples = step_input.reshape((batch_size, beam_size, 1))
        for i in range(self._max_length):
            log_probs, new_states = self._decoder(step_input, states)
            vocab_size_nd = mx.nd.array([log_probs.shape[1]], ctx=ctx)
            batch_shift_nd = mx.nd.arange(0, batch_size * beam_size, beam_size, ctx=ctx)
            step_nd = mx.nd.array([i + 1], ctx=ctx)
            samples, valid_length, scores, chosen_word_ids, beam_alive_mask, states = \
                self._updater(samples, valid_length, log_probs, scores, step_nd, beam_alive_mask,
                              new_states, vocab_size_nd, batch_shift_nd)
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


class HybridBeamSearchSampler(HybridBlock):
    r"""Draw samples from the decoder by beam search.

    Parameters
    ----------
    batch_size : int
        The batch size.
    beam_size : int
        The beam size.
    decoder : callable, must be hybridizable
        Function of the one-step-ahead decoder, should have the form::

            log_probs, new_states = decoder(step_input, states)

        The log_probs, input should follow these rules:

        - step_input has shape (batch_size,),
        - log_probs has shape (batch_size, V),
        - states and new_states have the same structure and the leading
          dimension of the inner NDArrays is the batch dimension.
    eos_id : int
        Id of the EOS token. No other elements will be appended to the sample if it reaches eos_id.
    scorer : BeamSearchScorer, default BeamSearchScorer(alpha=1.0, K=5), must be hybridizable
        The score function used in beam search.
    max_length : int, default 100
        The maximum search length.
    vocab_size : int, default None, meaning `decoder._vocab_size`
        The vocabulary size
    """
    def __init__(self, batch_size, beam_size, decoder, eos_id,
                 scorer=BeamSearchScorer(alpha=1.0, K=5),
                 max_length=100, vocab_size=None,
                 prefix=None, params=None):
        super(HybridBeamSearchSampler, self).__init__(prefix, params)
        self._batch_size = batch_size
        self._beam_size = beam_size
        assert beam_size > 0,\
            'beam_size must be larger than 0. Received beam_size={}'.format(beam_size)
        self._decoder = decoder
        self._eos_id = eos_id
        assert eos_id >= 0, 'eos_id cannot be negative! Received eos_id={}'.format(eos_id)
        self._max_length = max_length
        self._scorer = scorer
        self._state_info_func = getattr(decoder, 'state_info', lambda _=None: None)
        self._updater = _BeamSearchStepUpdate(beam_size=beam_size, eos_id=eos_id, scorer=scorer,
                                              single_step=True, state_info=self._state_info_func())
        self._updater.hybridize()
        self._vocab_size = vocab_size or getattr(decoder, '_vocab_size', None)
        assert self._vocab_size is not None,\
            'Please provide vocab_size or define decoder._vocab_size'
        assert not hasattr(decoder, '_vocab_size') or decoder._vocab_size == self._vocab_size, \
            'Provided vocab_size={} is not equal to decoder._vocab_size={}'\
            .format(self._vocab_size, decoder._vocab_size)

    def hybrid_forward(self, F, inputs, states):   # pylint: disable=arguments-differ
        """Sample by beam search.

        Parameters
        ----------
        F
        inputs : NDArray or Symbol
            The initial input of the decoder. Shape is (batch_size,).
        states : Object that contains NDArrays or Symbols
            The initial states of the decoder.
        Returns
        -------
        samples : NDArray or Symbol
            Samples draw by beam search. Shape (batch_size, beam_size, length). dtype is int32.
        scores : NDArray or Symbol
            Scores of the samples. Shape (batch_size, beam_size). We make sure that scores[i, :] are
            in descending order.
        valid_length : NDArray or Symbol
            The valid length of the samples. Shape (batch_size, beam_size). dtype will be int32.
        """
        batch_size = self._batch_size
        beam_size = self._beam_size
        vocab_size = self._vocab_size
        # Tile the states and inputs to have shape (batch_size * beam_size, ...)
        state_info = self._state_info_func(batch_size)
        step_input = _expand_to_beam_size(inputs, beam_size=beam_size, batch_size=batch_size)
        states = _expand_to_beam_size(states, beam_size=beam_size, batch_size=batch_size,
                                      state_info=state_info)
        state_structure, states = _extract_and_flatten_nested_structure(states)
        if beam_size == 1:
            init_scores = F.zeros(shape=(batch_size, 1))
        else:
            init_scores = F.concat(
                F.zeros(shape=(batch_size, 1)),
                F.full(shape=(batch_size, beam_size - 1), val=LARGE_NEGATIVE_FLOAT),
                dim=1,
            )
        vocab_size = F.full(shape=(1, ), val=vocab_size)
        batch_shift = F.arange(0, batch_size * beam_size, beam_size)

        def _loop_cond(_i, _samples, _indices, _step_input, _valid_length, _scores, \
            beam_alive_mask, *_states):
            return F.sum(beam_alive_mask) > 0

        def _loop_func(i, samples, indices, step_input, valid_length, scores, \
            beam_alive_mask, *states):
            log_probs, new_states = self._decoder(
                step_input, _reconstruct_flattened_structure(state_structure, states))
            step = i + 1
            new_samples, new_valid_length, new_scores, \
                chosen_word_ids, new_beam_alive_mask, new_new_states = \
                self._updater(samples, valid_length, log_probs, scores, step, beam_alive_mask,
                              _extract_and_flatten_nested_structure(new_states)[-1],
                              vocab_size, batch_shift)
            new_step_input = F.relu(chosen_word_ids).reshape((-1,))
            # We are doing `new_indices = indices[1 : ] + indices[ : 1]`
            new_indices = F.concat(
                indices.slice_axis(axis=0, begin=1, end=None),
                indices.slice_axis(axis=0, begin=0, end=1),
                dim=0,
            )
            return [], (step, new_samples, new_indices, new_step_input, new_valid_length, \
                   new_scores, new_beam_alive_mask) + tuple(new_new_states)

        _, pad_samples, indices, _, new_valid_length, new_scores, new_beam_alive_mask = \
            F.contrib.while_loop(
                cond=_loop_cond, func=_loop_func, max_iterations=self._max_length,
                loop_vars=(
                    F.zeros(shape=(1, )),                                       # i
                    F.zeros(shape=(batch_size, beam_size, self._max_length)),   # samples
                    F.arange(start=0, stop=self._max_length),                   # indices
                    step_input,                                                 # step_input
                    F.ones(shape=(batch_size, beam_size)),                      # valid_length
                    init_scores,                                                # scores
                    F.ones(shape=(batch_size, beam_size)),                      # beam_alive_mask
                ) + tuple(states)
            )[1][:7]                                                            # I hate Python 2
        samples = pad_samples.take(indices, axis=2)

        def _then_func():
            new_samples = F.concat(
                step_input.reshape((batch_size, beam_size, 1)),
                samples,
                F.full(shape=(batch_size, beam_size, 1), val=-1),
                dim=2)
            new_new_valid_length = new_valid_length
            return new_samples, new_new_valid_length

        def _else_func():
            final_word = F.where(new_beam_alive_mask,
                                 F.full(shape=(batch_size, beam_size), val=self._eos_id),
                                 F.full(shape=(batch_size, beam_size), val=-1))
            new_samples = F.concat(
                step_input.reshape((batch_size, beam_size, 1)),
                samples,
                final_word.reshape((0, 0, 1)),
                dim=2)
            new_new_valid_length = new_valid_length + new_beam_alive_mask
            return new_samples, new_new_valid_length

        new_samples, new_new_valid_length = \
            F.contrib.cond(F.sum(new_beam_alive_mask) == 0, _then_func, _else_func)
        return F.round(new_samples).astype(np.int32),\
               new_scores,\
               F.round(new_new_valid_length).astype(np.int32)
