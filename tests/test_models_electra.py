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
        electra_model = ElectraModel.from_cfg(cfg)
        electra_model.load_parameters(backbone_params_path)

        electra_disc_model = ElectraDiscriminator(cfg)
        electra_disc_model.load_parameters(disc_params_path)
        electra_disc_model = ElectraDiscriminator(cfg)
        electra_disc_model.backbone_model.load_parameters(backbone_params_path)

        gen_cfg = get_generator_cfg(cfg)
        word_embed_params = electra_disc_model.backbone_model.word_embed.collect_params()
        token_type_embed_params = electra_disc_model.backbone_model.token_pos_embed.collect_params()
        token_pos_embed_params = electra_disc_model.backbone_model.token_pos_embed.collect_params()
        embed_layer_norm_params = electra_disc_model.backbone_model.embed_layer_norm.collect_params()
        electra_gen_model = ElectraGenerator(gen_cfg,
                            tied_embeddings=True,
                            word_embed_params=word_embed_params,
                            token_type_embed_params=token_type_embed_params,
                            token_pos_embed_params=token_pos_embed_params,
                            embed_layer_norm_params=embed_layer_norm_params,
                            )
        electra_gen_model.load_parameters(gen_params_path)
        electra_gen_model = ElectraGenerator(cfg, tied_embeddings=False)
        electra_gen_model.backbone_model.load_parameters(backbone_params_path)
