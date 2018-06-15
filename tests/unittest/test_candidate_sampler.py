import mxnet as mx
import pytest

from gluonnlp.data.candidate_sampler import UnigramCandidateSampler


@pytest.mark.seed(1)
def test_unigram_candidate_sampler():
    N = 1000
    sampler = UnigramCandidateSampler(mx.nd.arange(N))
    assert all(mx.nd.array([691, 235, 908]) == sampler(3))
