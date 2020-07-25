import pytest
import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
import tempfile
from gluonnlp.models.electra import ElectraModel, ElectraDiscriminator, ElectraGenerator,\
    list_pretrained_electra, get_pretrained_electra, get_generator_cfg
mx.npx.set_np()


@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_electra())
def test_bert_get_pretrained(model_name):
    assert len(list_pretrained_electra()) > 0
    with tempfile.TemporaryDirectory() as root:
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
