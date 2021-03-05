import torch as th
import numpy as np
from gluonnlp.torch.layers import SinusoidalPositionalEmbedding


def test_sinusoidal_pos_embed():
    embed1 = SinusoidalPositionalEmbedding(128, learnable=False)
    embed2 = SinusoidalPositionalEmbedding(128, learnable=True)
    assert len([(name, param) for name, param in embed1.named_parameters()
                if param.requires_grad]) == 0
    assert len([(name, param) for name, param in embed2.named_parameters()
                if param.requires_grad]) == 1
    inputs = th.randint(0, 128, (8, 4))
    np.testing.assert_allclose(embed1(inputs).detach().cpu().numpy(),
                               embed2(inputs).detach().cpu().numpy(), 1E-3, 1E-3)
