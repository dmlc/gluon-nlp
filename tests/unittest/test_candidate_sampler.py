import numpy as np
import mxnet as mx
import pytest

from gluonnlp.data import candidate_sampler as cs


@pytest.mark.seed(1)
def test_unigram_candidate_sampler(hybridize):
    N = 1000
    sampler = cs.UnigramCandidateSampler(mx.nd.arange(N), shape=(3, ))
    sampler.initialize()
    if hybridize:
        sampler.hybridize()
    sampled = sampler(mx.nd.ones(3))
    print(sampled.asnumpy())
    assert np.all([729, 594, 690] == sampled.asnumpy())
