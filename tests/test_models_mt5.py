import pytest
import tempfile

from gluonnlp.models.mt5 import (
    MT5Model, MT5Inference, mt5_cfg_reg, list_pretrained_mt5, get_pretrained_mt5
)
from test_models_t5 import test_t5_model, test_t5_inference, test_t5_get_pretrained


def test_list_pretrained_mt5(): 
    assert len(list_pretrained_mt5()) > 0


@pytest.mark.parametrize('cfg_key', mt5_cfg_reg.list_keys())
def test_mt5_model_and_inference(cfg_key, ctx): 
    # since MT5Model, MT5Inference simply inherits the T5Model, T5Inference, 
    # we just want to make sure the model can be properly loaded, and leave 
    # the correctness tests to test_model_t5.py
    with ctx: 
        cfg = MT5Model.get_cfg(cfg_key)
        if cfg_key != 'google_mt5_small': 
            cfg.defrost()
            cfg.MODEL.vocab_size = 256
            cfg.MODEL.d_model = 128
            cfg.MODEL.d_ff = 512
            cfg.MODEL.num_layers = 2
            cfg.MODEL.num_heads = 4
            cfg.freeze()
        mt5_model = MT5Model.from_cfg(cfg)
        mt5_model.initialize()
        mt5_model.hybridize()
        if cfg_key == 'google_mt5_small': 
            inference_model = MT5Inference(mt5_model)
            inference_model.hybridize()


@pytest.mark.slow
@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', ['google_mt5_small'])# list_pretrained_mt5())
def test_mt5_get_pretrained(model_name, ctx): 
    assert len(list_pretrained_mt5()) > 0
    with tempfile.TemporaryDirectory() as root, ctx: 
        cfg, tokenizer, backbone_params_path, _ = get_pretrained_mt5(model_name)
        assert cfg.MODEL.vocab_size >= len(tokenizer.vocab)
        mt5_model = MT5Model.from_cfg(cfg)
        mt5_model.load_parameters(backbone_params_path)
        mt5_inference_model = MT5Inference(mt5_model)
    