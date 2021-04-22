__all__ = ['select_vectors_by_position', 'add_vectors_by_position',
           'update_vectors_by_position',
           'gumbel_softmax', 'trunc_gumbel',
           'relative_position_bucket',
           'l2_normalize']

import mxnet as mx
from mxnet import np, npx
import math
from mxnet import use_np


@use_np
def select_vectors_by_position(data, positions):
    """Select each batch with the given positions.

    Once advanced indexing can be hybridized, we can revise the implementation.

    out[i, j, ...] = data[i, positions[i, j], ...]

    Parameters
    ----------
    data
        Input tensor of contextualized token embeddings
        Shape (batch_size, seq_length, ...)
    positions
        Input tensor of the positions.
        Shape (batch_size, num_sel_positions).
        For each sample in the batch, the values in this tensor must not exceed
        the length of the sequence.

    Returns
    -------
    out
        The selection result.
        Shape (batch_size, num_sel_positions, ...)
    """
    # Here, we use gather_nd to select the output from data:
    # Need to compute
    #   out[i, j, :] = in[i, masked_position[i, j], :]
    # Thus, construct a indices with shape [2, batch_size, num_masked_position], where
    #     indices[0, i, j] = i
    #     indices[1, i, j] = masked_position[i, j]
    # Then, out = gather_nd(in, indices)
    positions = positions.astype(np.int32)
    # batch_idx.shape = (batch_size, 1) as [[0], [1], [2], ...]
    batch_idx = np.expand_dims(npx.arange_like(positions, axis=0),
                                 axis=1).astype(np.int32)
    batch_idx = batch_idx + np.zeros_like(positions)
    indices = np.stack([batch_idx, positions])
    # TODO(sxjscience) We can revise the implementation to advanced indexing
    #  once the bug in MXNet is solved:
    #  https://github.com/apache/incubator-mxnet/issues/18919
    out = npx.gather_nd(data, indices)
    return out


@use_np
def add_vectors_by_position(data, increment, positions):
    """Scatter each batch with the given positions.

    data[i, positions[i, j], ...] += increment[i, j, ...]

    Parameters
    ----------
    data
        Input tensor of the array to be updated.
        Shape (batch_size, seq_length, ...)
    increment
        Input tensor of token ids
        Shape (batch_size, num_disp_position, ...)
    positions
        Input tensor of the positions.
        Shape (batch_size, num_disp_position).
        For each sample in the batch, the values in this tensor must not exceed
        the length of the sequence.

    Returns
    -------
    out
        The updated result.
        Shape (batch_size, seq_length, ...)
    """
    # Here, we use index_add to disperse the output from data:
    # Need to compute
    #   out[i, masked_position[i, j], :] = in[i, j, :]
    # Thus, construct an indices with shape [2, batch_size * num_masked_position], where
    #     indices[0, i * num_masked_position + j] = i
    #     indices[1, i * num_masked_position + j] = masked_position[i, j]
    # And convert data to the shape of the (batch_size * num_masked_position, )
    # Then, out = npx.index_add(data, indices, increment)
    positions = positions.astype(np.int32)
    # batch_idx.shape = (batch_size, 1) as [[0], [1], [2], ...]
    batch_idx = np.expand_dims(npx.arange_like(positions, axis=0),
                                 axis=1).astype(np.int32)
    batch_idx = batch_idx + np.zeros_like(positions)
    indices = np.stack([batch_idx.reshape((-1,)), positions.reshape((-1,))])
    out = npx.index_add(data, indices, npx.reshape(increment, (-5, -4)))
    return out


@use_np
def update_vectors_by_position(data, val, positions):
    """
    Update each batch with the given positions. Considered as a reversed process of
    "select_vectors_by_position", this is an operator similar to "add_vectors_by_position"
    that updates the results instead of adding.

    data[i, positions[i, j], :] = val[i, j, :]

    Parameters
    ----------
    data
        Input tensor of the array to be updated.
        Shape (batch_size, seq_length)
    val
        Input tensor of token ids
        Shape (batch_size, num_disp_position)
    positions
        Input tensor of the positions.
        Shape (batch_size, num_disp_position).
        For each sample in the batch, the values in this tensor must not exceed
        the length of the sequence.

    Returns
    -------
    out
        The updated result.
        Shape (batch_size, seq_length)
    """
    positions = positions.astype(np.int32)
    # batch_idx.shape = (batch_size, 1) as [[0], [1], [2], ...]
    batch_idx = np.expand_dims(npx.arange_like(positions, axis=0),
                                 axis=1).astype(np.int32)
    batch_idx = batch_idx + np.zeros_like(positions)
    indices = np.stack([batch_idx.reshape((-1,)), positions.reshape((-1,))])

    out = npx.index_update(data, indices, npx.reshape(val, (-5, -4)))
    return out


@use_np
def gumbel_softmax(logits, temperature: float = 1.0, eps: float = 1E-10,
                   hard=True, use_np_gumbel: bool = True):
    r"""Perform the gumbel-softmax trick to generate differentiable one-hot vectors from the input
    logits.

    Here, the gumbel distribution is

    Gumbel(\alpha) = -log (-log U) + \log \alpha, in which U is the uniform(0, 1) distribution.

    A nice property of Gumbel is:

    \argmax({Gumbel(\alpha_i)}) \sim multinomial(\alpha_i)

    The Gumbel-Softmax trick is to use the softmax + straight-through estimator to produce
    one-hot vectors that represent the sampling result.

    References:

        1. https://en.wikipedia.org/wiki/Gumbel_distribution
        2. [ICLR2017] Categorical Reparameterization with Gumbel-Softmax

    Parameters
    ----------
    logits
        Logits. Shape (..., V)
    temperature
        The temperature that controls the
    eps
        The eps for stability of gradient
    hard
        Whether to use the straight-through estimator to produce one-hot vectors.
    use_np_gumbel
        Whether to use the random.gumble operator

    Returns
    -------
    ret
        The returned output. Shape (..., V)
    """
    # TODO(sxjscience) Investigate the impact of random.gumbel:
    #  Actually, random.gumble has no eps and may have problem in calculating the gradient.
    if use_np_gumbel:
        gumbels = np.random.gumbel(np.zeros_like(logits))
    else:
        u = np.random.uniform(np.zeros_like(logits), 1)
        gumbels = -np.log(-np.log(u + eps) + eps)
    y = npx.softmax((gumbels + logits) / temperature, axis=-1)
    if hard:
        y_hard = np.max(y, axis=-1, keepdims=True) == y
        y_hard = npx.stop_gradient(y_hard - y) + y
        return y_hard
    else:
        return y


def trunc_gumbel(logits, truncation):
    """Sample from the TruncGumbel distribution.

    The cumulative density function (CDF) of the Truncated Gumbel distribution is defined as

    TruncGumbel(\alpha, truncation) \prop max(Gumbel(\alpha), truncation)

    To sample from the distribution, we can use the CDF inversion technique.

    References:

        1. [NIPS2014] A* Sampling, https://papers.nips.cc/paper/5449-a-sampling.pdf
        2. https://cmaddis.github.io/gumbel-machinery

    Parameters
    ----------
    logits
        The logits. Shape (...,)
    truncation
        The truncation. Shape (...,)

    Returns
    -------
    samples
        Samples from the TruncGumbel(logits, truncation)
        Shape (...,)
    """
    gumbels = np.random.gumbel(np.zeros_like(logits)) + logits
    return -np.log(np.exp(-gumbels) + np.exp(-truncation))


def relative_position_bucket(relative_position,
                             bidirectional: bool = True,
                             num_buckets: int = 32,
                             max_distance: int = 128):
    """Map the relative position to buckets. The implementation is consistent with that
    in [mesh_tensorflow](https://github.com/tensorflow/mesh/blob/c59988047e49b4d2af05603e3170724cdbadc467/mesh_tensorflow/transformer/transformer_layers.py#L595-L637)
    where relative position is defined as `mem_i - query_j`. Thus, a positive value indicates 
    that the memory slot is in a later timestamp than the query slot. 

    After handling the bidirectional case (see below), the implementation uses the first half 
    of buckets to store exact differences and the second half to store the differences after 
    a logrithmic transformation. 

    Parameters
    ----------
    relative_position
        Shape (...,)
    bidirectional
        Whether we are dealing with bidirectional attention.
        If it's bidirectional, positive shifts are mapped to [0, num_buckets // 2), 
        and negative shifts are mapped to [num_buckets // 2, num_buckets). 
    num_buckets
        The number of buckets.
    max_distance
        Maximum distance. Positions that fall outside of 'max_distance' will be trimmed.

    Returns
    -------
    buckets
        Shape (...,).
        It has the same shape as the `relative_position`. It will have int32 type.
    """
    ret = 0
    relative_position = -relative_position
    if bidirectional:
        assert num_buckets % 2 == 0, 'When bidirectional is True, the number of buckets must be ' \
                                     'divisible by 2.'
        num_buckets //= 2
        ret = ret + (relative_position < 0).astype(np.int32) * num_buckets
        relative_position = np.abs(relative_position)
    else:
        # Clip all the negative values to 0
        relative_position = np.clip(relative_position, a_min=0, a_max=None)
    # Now, the relative_position is in the range [0, inf)

    # Half of the buckets deal with the exact increments,
    # i.e., 0, 1, 2, ..., max_exact - 1, where max_exact = num_buckets // 2
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to
    # max_distance
    val_if_large = max_exact + (
            np.log(relative_position.astype(np.float32) / max_exact)
            / math.log(max_distance / max_exact) * (num_buckets - max_exact)).astype(np.int32)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)
    ret = ret + np.where(is_small, relative_position, val_if_large)
    return ret


def l2_normalize(data, axis=-1, eps=1e-6):
    """Normalize the data by L2 normalization.

    Parameters
    ----------
    data
        The input data
    axis
        The axis that we should perform l2 normalization
    eps
        The epsilon value

    Returns
    -------
    ret
        The returned output
    """
    ret = data / (np.linalg.norm(data, axis=axis, keepdims=True) + eps)
    return ret
