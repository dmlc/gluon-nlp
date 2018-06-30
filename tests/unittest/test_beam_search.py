import functools

import mxnet as mx
import numpy as np
import pytest
from mxnet.gluon import Block, HybridBlock, nn
from mxnet.gluon.rnn import RNNCell, RNN
from numpy.testing import assert_allclose

from gluonnlp.model import BeamSearchSampler, BeamSearchScorer


def test_beam_search_score():
    batch_size = 2
    for length in [1, 5, 10]:
        for alpha, K in [(0.0, 1.0), (1.0, 5.0)]:
            scorer = BeamSearchScorer(alpha=alpha, K=K)
            scorer.hybridize()
            sum_log_probs = mx.nd.zeros((batch_size,))
            scores = mx.nd.zeros((batch_size,))
            for step in range(1, length + 1):
                log_probs = mx.nd.random.normal(0, 1, (batch_size, 1))
                sum_log_probs += log_probs[:, 0]
                scores = scorer(log_probs, scores, mx.nd.array([step]))[:, 0]
            lp = (K + length) ** alpha / (K + 1) ** alpha
            assert_allclose(scores.asnumpy(), sum_log_probs.asnumpy() / lp, 1E-5, 1E-5)


@pytest.mark.seed(1)
def test_beam_search():
    def _get_new_states(states, state_info, sel_beam_ids):
        assert not state_info or isinstance(state_info, (type(states), dict)), \
                'states and state_info don\'t match'
        if isinstance(states, list):
            if not state_info:
                state_info = [None] * len(states)
            return [_get_new_states(s, si, sel_beam_ids)
                    for s, si in zip(states, state_info)]
        elif isinstance(states, tuple):
            if not state_info:
                state_info = [None] * len(states)
                state_info = tuple(state_info)
            return tuple(_get_new_states(s, si, sel_beam_ids)
                         for s, si in zip(states, state_info))
        elif isinstance(states, dict):
            if not state_info:
                state_info = {k: None for k in states.keys()}
            return {k: _get_new_states(states[k], state_info[k], sel_beam_ids)
                    for k in states}
        elif isinstance(states, mx.nd.NDArray):
            updated_states = []
            if not state_info:
                batch_axis = 0
            else:
                batch_axis = state_info['__layout__'].find('N')
            if batch_axis != 0:
                states = states.swapaxes(0, batch_axis)
            for beam_id in sel_beam_ids:
                updated_states.append(states[beam_id])
            states = mx.nd.stack(*updated_states, axis=batch_axis)
            return states
        else:
            raise NotImplementedError

    def _fetch_step_states(states, state_info, batch_id, beam_size):
        assert not state_info or isinstance(state_info, (type(states), dict)), \
                'states and state_info don\'t match'
        if isinstance(states, list):
            if not state_info:
                state_info = [None] * len(states)
            return [_fetch_step_states(s, si, batch_id, beam_size)
                    for s, si in zip(states, state_info)]
        elif isinstance(states, tuple):
            if not state_info:
                state_info = [None] * len(states)
                state_info = tuple(state_info)
            return tuple(_fetch_step_states(s, si, batch_id, beam_size)
                         for s, si in zip(states, state_info))
        elif isinstance(states, dict):
            if not state_info:
                state_info = {k: None for k in states.keys()}
            return {k: _fetch_step_states(states[k], state_info[k], batch_id, beam_size)
                    for k in states}
        elif isinstance(states, mx.nd.NDArray):
            if not state_info:
                batch_axis = 0
            else:
                batch_axis = state_info['__layout__'].find('N')
            if batch_axis != 0:
                states = states.swapaxes(0, batch_axis)
            states = mx.nd.broadcast_axes(states[batch_id:(batch_id + 1)], axis=0, size=beam_size)
            if batch_axis != 0:
                states = states.swapaxes(0, batch_axis)
            return states
        else:
            raise NotImplementedError

    def _npy_beam_search(decoder, scorer, inputs, states, eos_id, beam_size, max_length):
        inputs = np.array([inputs for _ in range(beam_size)], dtype='float32')
        scores = np.array([0.0] + [-1e18] * (beam_size - 1), dtype='float32')
        samples = np.expand_dims(inputs, axis=1)
        beam_done = np.zeros(shape=(beam_size,), dtype='float32')
        if hasattr(decoder, 'state_info'):
            state_info = decoder.state_info()
        else:
            state_info = None
        for step in range(max_length):
            log_probs, states = decoder(mx.nd.array(inputs), states)
            vocab_num = log_probs.shape[1]
            candidate_scores = scorer(log_probs, mx.nd.array(scores),
                                      mx.nd.array([step + 1])).asnumpy()
            beam_done_inds = np.where(beam_done)[0]
            if len(beam_done_inds) > 0:
                candidate_scores[beam_done_inds, :] = -1e18
                finished_scores = scores[beam_done_inds]
                candidate_scores = np.concatenate(
                    (candidate_scores.reshape((-1,)), finished_scores), axis=0)
            else:
                candidate_scores = candidate_scores.reshape((-1,))
            indices = candidate_scores.argsort()[::-1][:beam_size]
            sel_words = []
            sel_beam_ids = []
            new_scores = candidate_scores[indices]
            for ind in indices:
                if ind < beam_size * vocab_num:
                    sel_words.append(ind % vocab_num)
                    sel_beam_ids.append(ind // vocab_num)
                else:
                    sel_words.append(-1)
                    sel_beam_ids.append(beam_done_inds[ind - beam_size * vocab_num])
            states = _get_new_states(states, state_info, sel_beam_ids)
            samples = np.concatenate((samples[sel_beam_ids, :],
                                      np.expand_dims(np.array(sel_words), axis=1)), axis=1)
            beam_done = np.logical_or(beam_done[sel_beam_ids], (np.array(sel_words) == eos_id))
            scores = new_scores
            inputs = [0 if ele < 0 else ele for ele in sel_words]
            if beam_done.all():
                return samples
        concat_val = - np.ones((beam_size,), dtype='float32') * beam_done + (1 - beam_done) * np.ones(
            (beam_size,), dtype='float32') * eos_id
        samples = np.concatenate((samples, np.expand_dims(concat_val, axis=1)), axis=1)
        return samples

    HIDDEN_SIZE = 2
    class RNNDecoder(HybridBlock):
        def __init__(self, vocab_num, hidden_size, prefix=None, params=None):
            super(RNNDecoder, self).__init__(prefix=prefix, params=params)
            self._vocab_num = vocab_num
            with self.name_scope():
                self._embed = nn.Embedding(input_dim=vocab_num, output_dim=hidden_size)
                self._rnn = RNNCell(hidden_size=hidden_size)
                self._map_to_vocab = nn.Dense(vocab_num)

        def begin_state(self, batch_size):
            return self._rnn.begin_state(batch_size=batch_size,
                                         func=functools.partial(mx.random.normal, loc=0, scale=1))

        def hybrid_forward(self, F, inputs, states):
            out, states = self._rnn(self._embed(inputs), states)
            log_probs = self._map_to_vocab(out)  # In real-life, we should add a log_softmax after that.
            return log_probs, states

    class RNNDecoder2(HybridBlock):
        def __init__(self, vocab_num, hidden_size, prefix=None, params=None, use_tuple=False):
            super(RNNDecoder2, self).__init__(prefix=prefix, params=params)
            self._vocab_num = vocab_num
            self._use_tuple = use_tuple
            with self.name_scope():
                self._embed = nn.Embedding(input_dim=vocab_num, output_dim=hidden_size)
                self._rnn1 = RNNCell(hidden_size=hidden_size)
                self._rnn2 = RNNCell(hidden_size=hidden_size)
                self._map_to_vocab = nn.Dense(vocab_num)

        def begin_state(self, batch_size):
            ret = [self._rnn1.begin_state(batch_size=batch_size,
                                           func=functools.partial(mx.random.normal, loc=0, scale=1)),
                    self._rnn2.begin_state(batch_size=batch_size,
                                           func=functools.partial(mx.random.normal, loc=0, scale=1))]
            if self._use_tuple:
                return tuple(ret)
            else:
                return ret

        def hybrid_forward(self, F, inputs, states):
            if self._use_tuple:
                states1, states2 = states
            else:
                [states1, states2] = states
            out1, states1 = self._rnn1(self._embed(inputs), states1)
            out2, states2 = self._rnn2(out1, states2)
            log_probs = self._map_to_vocab(out2)  # In real-life, we should add a log_softmax after that.
            if self._use_tuple:
                states = (states1, states2)
            else:
                states = [states1, states2]
            return log_probs, states

    class RNNLayerDecoder(Block):
        def __init__(self, vocab_num, hidden_size, prefix=None, params=None):
            super(RNNLayerDecoder, self).__init__(prefix=prefix, params=params)
            self._vocab_num = vocab_num
            with self.name_scope():
                self._embed = nn.Embedding(input_dim=vocab_num, output_dim=hidden_size)
                self._rnn = RNN(hidden_size=hidden_size, num_layers=1, activation='tanh')
                self._map_to_vocab = nn.Dense(vocab_num, flatten=False)

        def begin_state(self, batch_size):
            return self._rnn.begin_state(batch_size=batch_size,
                                         func=functools.partial(mx.random.normal, loc=0, scale=5))

        def state_info(self, *args, **kwargs):
            return self._rnn.state_info(*args, **kwargs)

        def forward(self, inputs, states):
            out, states = self._rnn(self._embed(inputs.expand_dims(0)), states)
            log_probs = self._map_to_vocab(out)[0].log_softmax()
            return log_probs, states

    # Begin Testing
    for vocab_num in [4, 8]:
        for decoder_fn in [RNNDecoder,
                           functools.partial(RNNDecoder2, use_tuple=False),
                           functools.partial(RNNDecoder2, use_tuple=True),
                           RNNLayerDecoder]:
            decoder = decoder_fn(vocab_num=vocab_num, hidden_size=HIDDEN_SIZE)
            decoder.hybridize()
            decoder.initialize()
            if hasattr(decoder, 'state_info'):
                state_info = decoder.state_info()
            else:
                state_info = None
            for beam_size, bos_id, eos_id, alpha, K in [(2, 1, 3, 0, 1.0),  (4, 2, 3, 1.0, 5.0)]:
                scorer = BeamSearchScorer(alpha=alpha, K=K)
                for max_length in [10, 20]:
                    sampler = BeamSearchSampler(beam_size=beam_size, decoder=decoder, eos_id=eos_id,
                                                scorer=scorer, max_length=max_length)
                    for batch_size in [1, 2, 5]:
                        print(type(decoder).__name__, beam_size, bos_id, eos_id, alpha, K, batch_size)
                        states = decoder.begin_state(batch_size)
                        inputs = mx.nd.full(shape=(batch_size,), val=bos_id)
                        samples, scores, valid_length = sampler(inputs, states)
                        samples = samples.asnumpy()
                        scores = scores.asnumpy()
                        valid_length = valid_length.asnumpy()
                        for i in range(batch_size):
                            max_beam_valid_length = int(np.round(valid_length[i].max()))
                            step_states = _fetch_step_states(states, state_info, i, beam_size)
                            step_input = bos_id
                            npy_samples = _npy_beam_search(decoder, scorer, step_input, step_states,
                                                           eos_id, beam_size, max_length)
                            selected_samples = samples[i, :, :max_beam_valid_length]
                            assert_allclose(npy_samples, selected_samples)
                            for j in range(beam_size):
                                assert(samples[i, j, valid_length[i, j] - 1] == 3.0)
                                if valid_length[i, j] < samples.shape[2]:
                                    assert((samples[i, j, valid_length[i, j]:] == -1.0).all())
