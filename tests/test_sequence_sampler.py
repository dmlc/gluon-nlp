import collections
import functools
import mxnet as mx
import numpy as np
import scipy
import pytest
from mxnet.gluon import nn, HybridBlock
from numpy.testing import assert_allclose
from gluonnlp.sequence_sampler import BeamSearchScorer, BeamSearchSampler
mx.npx.set_np()


@pytest.mark.parametrize('length', [False, True])
@pytest.mark.parametrize('alpha', [0.0, 1.0])
@pytest.mark.parametrize('K', [1.0, 5.0])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('vocab_size', [2, 5])
@pytest.mark.parametrize('from_logits', [False, True])
@pytest.mark.parametrize('hybridize', [False, True])
def test_beam_search_score(length, alpha, K, batch_size, vocab_size, from_logits, hybridize):
    scorer = BeamSearchScorer(alpha=alpha, K=K, from_logits=from_logits)
    if hybridize:
        scorer.hybridize()
    sum_log_probs = mx.np.zeros((batch_size,))
    scores = mx.np.zeros((batch_size,))
    for step in range(1, length + 1):
        if not from_logits:
            log_probs = np.random.normal(0, 1, (batch_size, vocab_size))
            log_probs = np.log((scipy.special.softmax(log_probs, axis=-1)))
        else:
            log_probs = np.random.uniform(-10, 0, (batch_size, vocab_size))
        log_probs = mx.np.array(log_probs, dtype=np.float32)
        sum_log_probs += log_probs[:, 0]
        scores = scorer(log_probs, scores, mx.np.array(step))[:, 0]
    lp = (K + length) ** alpha / (K + 1) ** alpha
    assert_allclose(scores.asnumpy(), sum_log_probs.asnumpy() / lp, 1E-5, 1E-5)


# TODO(sxjscience) Test for the state_batch_axis
@pytest.mark.parametrize('early_return', [False, True])
def test_beam_search(early_return):
    class SimpleStepDecoder(HybridBlock):
        def __init__(self, vocab_size=5, hidden_units=4, prefix=None, params=None):
            super(SimpleStepDecoder, self).__init__(prefix=prefix, params=params)
            self.x2h_map = nn.Embedding(input_dim=vocab_size, output_dim=hidden_units)
            self.h2h_map = nn.Dense(units=hidden_units, flatten=False)
            self.vocab_map = nn.Dense(units=vocab_size, flatten=False)

        @property
        def state_batch_axis(self):
            return 0

        def hybrid_forward(self, F, data, state):
            """

            Parameters
            ----------
            F
            data :
                (batch_size,)
            states :
                (batch_size, C)

            Returns
            -------
            out :
                (batch_size, vocab_size)
            new_state :
                (batch_size, C)
            """
            new_state = self.h2h_map(state)
            out = self.vocab_map(self.x2h_map(data) + new_state)
            return out, new_state

    vocab_size = 3
    batch_size = 2
    hidden_units = 3
    beam_size = 4
    step_decoder = SimpleStepDecoder(vocab_size, hidden_units)
    step_decoder.initialize()
    sampler = BeamSearchSampler(beam_size=4, decoder=step_decoder, eos_id=0, vocab_size=vocab_size,
                                max_length_b=100)
    states = mx.np.random.normal(0, 1, (batch_size, hidden_units))
    inputs = mx.np.random.randint(0, vocab_size, (batch_size,))
    samples, scores, valid_length = sampler(inputs, states, early_return=early_return)
    samples = samples.asnumpy()
    valid_length = valid_length.asnumpy()
    for i in range(batch_size):
        for j in range(beam_size):
            vl = valid_length[i, j]
            assert samples[i, j, vl - 1] == 0
            if vl < samples.shape[2]:
                assert (samples[i, j, vl:] == -1).all()
            assert (samples[i, :, 0] == inputs[i].asnumpy()).all()

