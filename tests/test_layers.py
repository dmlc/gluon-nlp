import pytest
import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
from gluonnlp.layers import\
    MultiHeadDense, PositionalEmbedding, \
    SinusoidalPositionalEmbedding, \
    LearnedPositionalEmbedding, \
    BucketPositionalEmbedding, \
    AdaptiveEmbedding, \
    ProjectedAdaptiveLogSoftmaxWithLoss, \
    get_activation, \
    get_norm_layer
from gluonnlp.op import relative_position_bucket
mx.npx.set_np()


def test_multi_head_dense():
    def _verify(num_heads, hybridize):
        layer = MultiHeadDense(32, num_heads)
        layer.initialize()
        if hybridize:
            layer.hybridize()
        in_data = mx.np.ones((5, 4, 10))
        out = layer(in_data)
        if not isinstance(num_heads, (list, tuple)):
            num_heads = (num_heads,)
        else:
            num_heads = tuple(num_heads)
        assert out.shape == (5,) + num_heads + (4, 32)
        out_data = out.asnumpy()
        weight_data = layer.weight.data().asnumpy()
        bias_data = layer.bias.data().asnumpy()
        gt_data = (in_data.asnumpy().dot(weight_data.T) + bias_data)\
            .reshape((5, 4, np.prod(num_heads), 32))
        gt_data = np.moveaxis(gt_data, -2, 1)
        gt_data = gt_data.reshape((5,) + num_heads + (4, 32))
        assert_allclose(gt_data, out_data, 1E-4, 1E-4)
    for parallel_num in [3, (2, 3), (3, 2, 3)]:
        for hybridize in [True, False]:
            _verify(parallel_num, hybridize)


def test_sinusoidal_positional_embedding():
    def _gt_sinusoidal_embedding(np_data, units):
        half_dim = units // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)
        emb = np.expand_dims(np_data.astype(np.float32), axis=-1) * emb
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
        if units % 2 == 1:
            # zero pad
            emb = np.concatenate([emb, np.expand_dims(np.zeros_like(np_data), axis=-1)], axis=-1)
        return emb
    for units in [31, 16]:
        for hybridize in [False, True]:
            pos_embed = SinusoidalPositionalEmbedding(units=units, dtype=np.float32)
            pos_embed.initialize(mx.init.Normal(0.01))
            if hybridize:
                pos_embed.hybridize()
            in_data = mx.np.array([100, 5, 20], dtype=np.int32)
            out = pos_embed(in_data)
            gt_out = _gt_sinusoidal_embedding(in_data.asnumpy(), units=units)
            assert_allclose(gt_out, out.asnumpy(), 1E-5, 1E-5)


def test_positional_embedding():
    pos_embed = PositionalEmbedding(method='sinusoidal', units=128)
    assert isinstance(pos_embed._embed, SinusoidalPositionalEmbedding)
    pos_embed = PositionalEmbedding(method='learned', units=128)
    assert isinstance(pos_embed._embed, LearnedPositionalEmbedding)


def test_get_activation():
    # Here we just test that the scripts are runnable. Should be revised to test for correctness
    for act_type in ['leaky', 'identity', 'elu', 'gelu', 'gelu(tanh)', 'gelu(sigmoid)',
                     'relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']:
        act = get_activation(act_type)
        act.hybridize()
        _ = act(mx.np.random.normal(0, 1, (10, 10)))


@pytest.mark.parametrize('vocab_size,cutoffs,div_val',
                         [[1000, 10, 1.5],
                          [1000, [5], 1.5],
                          [500, [20, 100], 1.5],
                          [1000, 5, 1.0]])
@pytest.mark.parametrize('embed_size', [128])
@pytest.mark.parametrize('units', [16])
def test_adaptive_embedding(vocab_size, cutoffs, embed_size, units, div_val):
    embed = AdaptiveEmbedding(vocab_size=vocab_size, embed_size=embed_size,
                              units=units, cutoffs=cutoffs, div_val=div_val)
    embed.initialize()
    embed.hybridize()
    # Test for parameter number
    estimated_param_num = 0
    if isinstance(cutoffs, int):
        cutoffs = [cutoffs]
    if div_val != 1.0:
        for i, (lhs, rhs) in enumerate(zip([0] + cutoffs, cutoffs + [vocab_size])):
            estimated_param_num += (rhs - lhs) * int(embed_size / div_val ** i)
            estimated_param_num += int(embed_size / div_val ** i) * units
        total_param_num = sum([np.prod(p.shape) for p in embed.collect_params().values()])
    else:
        estimated_param_num = vocab_size * embed_size + embed_size * units
        total_param_num = sum([np.prod(p.shape) for p in embed.collect_params().values()])
    assert total_param_num == estimated_param_num
    # Test for forward
    out = embed(mx.np.random.randint(0, vocab_size, 20))
    mx.npx.waitall()
    assert out.shape == (20, units)


@pytest.mark.parametrize('vocab_size,cutoffs,div_val',
                         [[1000, 10, 1.5],
                          [1000, [5], 1.5],
                          [500, [20, 100], 1.5],
                          [500, [20, 100], 1.],
                          [1000, 10, 1.0],
                          [1000, [], 1.0],
                          [1000, None, 1.0]])
@pytest.mark.parametrize('embed_size', [128])
@pytest.mark.parametrize('in_units', [16])
# TODO This test even passes without sharing the parameters. It needs to be improved.
def test_projected_adaptive_softmax(vocab_size, cutoffs, embed_size, in_units, div_val):
    layer = ProjectedAdaptiveLogSoftmaxWithLoss(vocab_size=vocab_size, cutoffs=cutoffs,
                                                embed_size=embed_size, in_units=in_units,
                                                div_val=div_val)
    layer.initialize()
    layer.hybridize()
    hidden = mx.np.random.normal(0, 1, (4, 4, 4, 16))
    target = mx.np.random.randint(0, vocab_size, (4, 4, 4,))
    out = layer(hidden, target)
    mx.npx.waitall()
    assert out.shape == (4, 4, 4)

    # Test for weight sharing
    embed_layer = AdaptiveEmbedding(vocab_size=vocab_size, cutoffs=cutoffs,
                                    units=in_units, embed_size=embed_size,
                                    div_val=div_val)
    layer_with_shared_proj = \
        ProjectedAdaptiveLogSoftmaxWithLoss(vocab_size=vocab_size,
                                            cutoffs=cutoffs,
                                            embed_size=embed_size,
                                            in_units=in_units,
                                            div_val=div_val)
    layer_with_shared_proj.share_parameters(embed_layer.collect_params('inter_proj'))
    layer_with_shared_embed = \
        ProjectedAdaptiveLogSoftmaxWithLoss(vocab_size=vocab_size,
                                            cutoffs=cutoffs,
                                            embed_size=embed_size,
                                            in_units=in_units,
                                            div_val=div_val)
    layer_with_shared_embed.share_parameters(embed_layer.collect_params('embed'))
    layer_with_shared_proj_embed = \
        ProjectedAdaptiveLogSoftmaxWithLoss(vocab_size=vocab_size,
                                            cutoffs=cutoffs,
                                            embed_size=embed_size,
                                            in_units=in_units,
                                            div_val=div_val)
    layer_with_shared_proj_embed.share_parameters(embed_layer.collect_params('(embed|inter_proj)'))
    embed_layer.initialize()
    embed_layer.hybridize()
    layer_with_shared_proj.initialize()
    layer_with_shared_proj.hybridize()
    layer_with_shared_embed.initialize()
    layer_with_shared_embed.hybridize()
    layer_with_shared_proj_embed.initialize()
    layer_with_shared_proj_embed.hybridize()

    hidden = mx.np.random.normal(0, 1, (4, 4, 4, 16))
    target = mx.np.random.randint(0, vocab_size, (4, 4, 4,))
    with mx.autograd.record():
        loss = ((hidden - embed_layer(target)) ** 2).sum()
        loss.backward()
    assert embed_layer(target).asnumpy().shape == hidden.shape

    embed_weights = {}
    embed_grads = {}
    proj_weights = {}
    proj_grads = {}
    for k, v in embed_layer.collect_params().items():
        if '_embed' in k:
            arr_id = int(k[-len('_weight') - 1])
            embed_weights[arr_id] = v.data()[0].asnumpy()
            embed_grads[arr_id] = v.grad()[0].asnumpy()
        elif '_inter_proj' in k:
            arr_id = int(k[-len('_weight') - 1])
            proj_weights[arr_id] = v.data()[0].asnumpy()
            proj_grads[arr_id] = v.grad()[0].asnumpy()

    # Check shared proj
    for k, v in layer_with_shared_proj.collect_params().items():
        if '_embed' in k and '_weight' in k:
            arr_id = int(k[-len('_weight') - 1])
            with pytest.raises(AssertionError):
                assert_allclose(v.data()[0].asnumpy(), embed_weights[arr_id])
        elif '_inter_proj' in k and '_weight' in k:
            arr_id = int(k[-len('_weight') - 1])
            assert_allclose(v.data()[0].asnumpy(), proj_weights[arr_id])
            assert_allclose(v.grad()[0].asnumpy(), proj_grads[arr_id])

    # Check shared embed
    for k, v in layer_with_shared_embed.collect_params().items():
        if '_embed' in k and '_weight' in k:
            arr_id = int(k[-len('_weight') - 1])
            assert_allclose(v.data()[0].asnumpy(), embed_weights[arr_id])
            assert_allclose(v.grad()[0].asnumpy(), embed_grads[arr_id])
        elif '_inter_proj' in k and '_weight' in k:
            arr_id = int(k[-len('_weight') - 1])
            with pytest.raises(AssertionError):
                assert_allclose(v.data()[0].asnumpy(), proj_weights[arr_id])

    # Check shared proj + shared embed
    for k, v in layer_with_shared_proj_embed.collect_params().items():
        if '_embed' in k and '_weight' in k:
            arr_id = int(k[-len('_weight') - 1])
            assert_allclose(v.data()[0].asnumpy(), embed_weights[arr_id])
            assert_allclose(v.grad()[0].asnumpy(), embed_grads[arr_id])
        elif '_inter_proj' in k and '_weight' in k:
            arr_id = int(k[-len('_weight') - 1])
            assert_allclose(v.data()[0].asnumpy(), proj_weights[arr_id])
            assert_allclose(v.grad()[0].asnumpy(), proj_grads[arr_id])


@pytest.mark.parametrize('units', [16])
@pytest.mark.parametrize('num_buckets', [32, 64])
@pytest.mark.parametrize('bidirectional', [True, False])
@pytest.mark.parametrize('max_distance', [128, 256])
@pytest.mark.seed(123)
def test_bucket_positional_embedding(units, num_buckets, bidirectional, max_distance):
    embed = BucketPositionalEmbedding(units=units, bidirectional=bidirectional,
                                      num_buckets=num_buckets, max_distance=max_distance)
    embed.initialize()
    relative_positions1 = mx.np.random.randint(-10000, 10000, (128, 25), dtype=np.int32)
    relative_positions2 = mx.np.random.randint(-10, 10, (128, 25), dtype=np.int32)
    relative_positions = mx.np.concatenate([relative_positions1, relative_positions2], axis=-1)
    buckets = relative_position_bucket(mx, relative_positions, bidirectional=bidirectional,
                                       num_buckets=num_buckets, max_distance=max_distance)
    out = embed(relative_positions)
    for i in range(num_buckets):
        cnt = (buckets == i).sum().asnumpy()
        if cnt > 1:
            assert_allclose(mx.np.linalg.norm(out[buckets == i].std(axis=0)).asnumpy(), 0,
                            1E-5, 1E-5)
    if bidirectional:
        assert mx.np.all(buckets[relative_positions < 0] >= num_buckets // 2).asnumpy()
    out_of_bound_cnt = buckets[relative_positions > max_distance].sum()
    if out_of_bound_cnt.asnumpy() > 0:
        assert buckets[relative_positions > max_distance].std().asnumpy() == 0


@pytest.mark.parametrize('normalization', ['layer_norm', 'no_norm', 'identity', 'batch_norm'])
def test_get_norm_layer(normalization, ctx):
    with ctx:
        norm_layer = get_norm_layer(normalization=normalization,
                                    in_channels=16)
        net = mx.gluon.nn.HybridSequential()
        net.add(mx.gluon.nn.Dense(16, in_units=16))
        net.add(norm_layer)
        net.add(mx.gluon.nn.Dense(16, in_units=16))
        net.hybridize()
        net.initialize()
        data_in = mx.np.random.normal(0, 1, (8, 16))
        out = net(data_in)
        out_np = out.asnumpy()
