import pytest
import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
import tempfile
from gluonnlp.models.albert import AlbertModel, AlbertForMLM, AlbertForPretrain,\
    list_pretrained_albert, get_pretrained_albert
mx.npx.set_np()


def get_test_cfg():
    vocab_size = 500
    num_token_types = 3
    num_layers = 3
    num_heads = 2
    units = 64
    hidden_size = 96
    hidden_dropout_prob = 0.0
    attention_dropout_prob = 0.0
    cfg = AlbertModel.get_cfg().clone()
    cfg.defrost()
    cfg.MODEL.vocab_size = vocab_size
    cfg.MODEL.num_token_types = num_token_types
    cfg.MODEL.units = units
    cfg.MODEL.hidden_size = hidden_size
    cfg.MODEL.num_heads = num_heads
    cfg.MODEL.num_layers = num_layers
    cfg.MODEL.hidden_dropout_prob = hidden_dropout_prob
    cfg.MODEL.attention_dropout_prob = attention_dropout_prob
    return cfg


def test_albert_backbone():
    batch_size = 3
    cfg = get_test_cfg()
    model = AlbertModel.from_cfg(cfg, use_pooler=True)
    model.initialize()
    model.hybridize(static_alloc=True, static_shape=True)
    for seq_length in [64, 96]:
        valid_length = mx.np.random.randint(seq_length // 2, seq_length, (batch_size,))
        inputs = mx.np.random.randint(0, cfg.MODEL.vocab_size, (batch_size, seq_length))
        token_types = mx.np.random.randint(0, cfg.MODEL.num_token_types, (batch_size, seq_length))
        contextual_embedding, pooled_out = model(inputs, token_types, valid_length)
        assert contextual_embedding.shape == (batch_size, seq_length, cfg.MODEL.units)
        assert pooled_out.shape == (batch_size, cfg.MODEL.units)
        # Ensure the embeddings that exceed valid_length are masked
        contextual_embedding_np = contextual_embedding.asnumpy()
        pooled_out_np = pooled_out.asnumpy()
        for i in range(batch_size):
            ele_valid_length = valid_length[i].asnumpy()
            assert_allclose(contextual_embedding_np[i, ele_valid_length:],
                            np.zeros_like(contextual_embedding_np[i, ele_valid_length:]),
                            1E-5, 1E-5)
        # Ensure that the content are correctly masked
        new_inputs = mx.np.concatenate([inputs, inputs[:, :5]], axis=-1)
        new_token_types = mx.np.concatenate([token_types, token_types[:, :5]], axis=-1)
        new_contextual_embedding, new_pooled_out = \
            model(new_inputs, new_token_types, valid_length)
        new_contextual_embedding_np = new_contextual_embedding.asnumpy()
        new_pooled_out_np = new_pooled_out.asnumpy()
        for i in range(batch_size):
            ele_valid_length = valid_length[i].asnumpy()
            assert_allclose(new_contextual_embedding_np[i, :ele_valid_length],
                            contextual_embedding_np[i, :ele_valid_length], 1E-5, 1E-5)
        assert_allclose(new_pooled_out_np, pooled_out_np, 1E-4, 1E-4)


def test_albert_for_mlm_model():
    batch_size = 3
    cfg = get_test_cfg()
    albert_mlm_model = AlbertForMLM(backbone_cfg=cfg)
    albert_mlm_model.initialize()
    albert_mlm_model.hybridize()
    num_mask = 16
    seq_length = 64
    inputs = mx.np.random.randint(0, cfg.MODEL.vocab_size, (batch_size, seq_length))
    token_types = mx.np.random.randint(0, cfg.MODEL.num_token_types, (batch_size, seq_length))
    valid_length = mx.np.random.randint(seq_length // 2, seq_length, (batch_size,))
    masked_positions = mx.np.random.randint(0, seq_length // 2, (batch_size, num_mask))
    _, _, mlm_scores = albert_mlm_model(inputs, token_types, valid_length, masked_positions)
    assert mlm_scores.shape == (batch_size, num_mask, cfg.MODEL.vocab_size)


def test_albert_for_pretrain_model():
    batch_size = 3
    cfg = get_test_cfg()
    albert_pretrain_model = AlbertForPretrain(backbone_cfg=cfg)
    albert_pretrain_model.initialize()
    albert_pretrain_model.hybridize()
    num_mask = 16
    seq_length = 64
    inputs = mx.np.random.randint(0, cfg.MODEL.vocab_size, (batch_size, seq_length))
    token_types = mx.np.random.randint(0, cfg.MODEL.num_token_types, (batch_size, seq_length))
    valid_length = mx.np.random.randint(seq_length // 2, seq_length, (batch_size,))
    masked_positions = mx.np.random.randint(0, seq_length // 2, (batch_size, num_mask))
    _, _, sop_score, mlm_scores = albert_pretrain_model(inputs, token_types, valid_length, masked_positions)
    assert mlm_scores.shape == (batch_size, num_mask, cfg.MODEL.vocab_size)
    assert sop_score.shape == (batch_size, 2)


def test_list_pretrained_albert():
    assert len(list_pretrained_albert()) > 0


@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_albert())
def test_albert_get_pretrained(model_name):
    assert len(list_pretrained_albert()) > 0
    with tempfile.TemporaryDirectory() as root:
        cfg, tokenizer, backbone_params_path, mlm_params_path =\
            get_pretrained_albert(model_name, root=root)
        albert_model = AlbertModel.from_cfg(cfg)
        albert_model.load_parameters(backbone_params_path)
        albert_mlm_model = AlbertForMLM(cfg)
        albert_mlm_model.load_parameters(mlm_params_path)
        # Just load the backbone
        albert_mlm_model = AlbertForMLM(cfg)
        albert_mlm_model.backbone_model.load_parameters(backbone_params_path)
