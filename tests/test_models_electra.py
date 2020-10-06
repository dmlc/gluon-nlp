import pytest
import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
import tempfile
from gluonnlp.models.electra import ElectraModel, ElectraDiscriminator,\
    ElectraGenerator,\
    list_pretrained_electra, get_pretrained_electra, get_generator_cfg
mx.npx.set_np()


def test_list_pretrained_electra():
    assert len(list_pretrained_electra()) > 0


def get_test_cfg():
    cfg = ElectraModel.get_cfg()
    cfg.defrost()
    cfg.MODEL.vocab_size = 100
    cfg.MODEL.units = 12 * 8
    cfg.MODEL.hidden_size = 128
    cfg.MODEL.num_heads = 2
    cfg.MODEL.num_layers = 2
    cfg.freeze()
    return cfg


@pytest.mark.parametrize('compute_layout', ['auto', 'NT', 'TN'])
def test_electra_model(compute_layout, ctx):
    with ctx:
        cfg = get_test_cfg()
        cfg.defrost()
        cfg.MODEL.compute_layout = compute_layout
        cfg.freeze()

        # Generate TN layout
        cfg_tn = cfg.clone()
        cfg_tn.defrost()
        cfg_tn.MODEL.layout = 'TN'
        cfg_tn.freeze()

        # Sample data
        batch_size = 4
        sequence_length = 16
        num_mask = 3
        inputs = mx.np.random.randint(0, 10, (batch_size, sequence_length))
        token_types = mx.np.random.randint(0, 2, (batch_size, sequence_length))
        valid_length = mx.np.random.randint(3, sequence_length, (batch_size,))
        masked_positions = mx.np.random.randint(0, 3, (batch_size, num_mask))

        electra_model = ElectraModel.from_cfg(cfg)
        electra_model.initialize()
        electra_model.hybridize()
        contextual_embedding, pooled_out = electra_model(inputs, token_types, valid_length)
        electra_model_tn = ElectraModel.from_cfg(cfg_tn)
        electra_model_tn.share_parameters(electra_model.collect_params())
        electra_model_tn.hybridize()
        contextual_embedding_tn, pooled_out_tn = electra_model_tn(inputs.T, token_types.T, valid_length)
        assert_allclose(contextual_embedding.asnumpy(),
                        np.swapaxes(contextual_embedding_tn.asnumpy(), 0, 1),
                        1E-4, 1E-4)
        assert_allclose(pooled_out.asnumpy(), pooled_out_tn.asnumpy(),
                        1E-4, 1E-4)


@pytest.mark.slow
@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_electra())
def test_electra_get_pretrained(model_name, ctx):
    assert len(list_pretrained_electra()) > 0
    with tempfile.TemporaryDirectory() as root, ctx:
        cfg, tokenizer, backbone_params_path, (disc_params_path, gen_params_path) =\
            get_pretrained_electra(model_name, root=root,
                                   load_backbone=True, load_disc=True, load_gen=True)
        assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
        electra_model = ElectraModel.from_cfg(cfg)
        electra_model.load_parameters(backbone_params_path)

        electra_disc_model = ElectraDiscriminator(cfg)
        electra_disc_model.load_parameters(disc_params_path)
        electra_disc_model = ElectraDiscriminator(cfg)
        electra_disc_model.backbone_model.load_parameters(backbone_params_path)

        gen_cfg = get_generator_cfg(cfg)
        electra_gen_model = ElectraGenerator(gen_cfg)
        electra_gen_model.load_parameters(gen_params_path)
        electra_gen_model.tie_embeddings(
            electra_disc_model.backbone_model.word_embed.collect_params(),
            electra_disc_model.backbone_model.token_type_embed.collect_params(),
            electra_disc_model.backbone_model.token_pos_embed.collect_params(),
            electra_disc_model.backbone_model.embed_layer_norm.collect_params())

        electra_gen_model = ElectraGenerator(cfg)
        electra_gen_model.backbone_model.load_parameters(backbone_params_path)
