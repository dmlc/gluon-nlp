import pytest
import mxnet as mx
import tempfile
from gluonnlp.models.bart import BartModel, \
    list_pretrained_bart, get_pretrained_bart, bart_cfg_reg


mx.npx.set_np()


def test_list_pretrained_bart():
    assert len(list_pretrained_bart()) > 0


@pytest.mark.slow
@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_bart())
def test_bart(model_name):
    # test from pretrained
    assert len(list_pretrained_bart()) > 0
    with tempfile.TemporaryDirectory() as root:
        cfg, tokenizer, params_path, _ =\
            get_pretrained_bart(model_name, load_backbone=True, root=root)
        assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
        # test standard bart encoder and decoder
        bart_model = BartModel.from_cfg(cfg)
        bart_model.load_parameters(params_path)
        # test bart encoder and decoder with pooler
        bart_model_with_pooler = BartModel.from_cfg(
            cfg, use_pooler=True, classifier_activation=False)
        bart_model_with_pooler.load_parameters(params_path)


def test_bart_cfg_registry():
    assert len(bart_cfg_reg.list_keys()) > 0


@pytest.mark.parametrize('cfg_key', bart_cfg_reg.list_keys())
def test_bart_cfg(cfg_key):
    cfg = BartModel.get_cfg(cfg_key)
    cfg.defrost()
    cfg.MODEL.vocab_size = 32
    cfg.freeze()
    model = BartModel.from_cfg(cfg)
    model.initialize()
    model.hybridize()
    cfg.defrost()
    cfg.MODEL.layout = 'TN'
    cfg.freeze()
    model_tn = BartModel.from_cfg(cfg)
    model_tn.share_parameters(model.collect_params())
    model_tn.hybridize()
    mx.npx.waitall()
