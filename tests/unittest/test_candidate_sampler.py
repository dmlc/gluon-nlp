import mxnet as mx
import pytest

from gluonnlp.data import candidate_sampler as cs


@pytest.mark.seed(1)
def test_unigram_candidate_sampler():
    N = 1000
    sampler = cs.UnigramCandidateSampler(mx.nd.arange(N))
    sampled = sampler(3)
    assert all(mx.nd.array([729, 593, 689]) == sampled)
