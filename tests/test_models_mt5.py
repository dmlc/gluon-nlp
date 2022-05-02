import pytest
import tempfile

from gluonnlp.models.mt5 import (
    MT5Model, MT5Inference, mt5_cfg_reg, list_pretrained_mt5, get_pretrained_mt5
)

def test_list_pretrained_mt5(): 
    assert len(list_pretrained_mt5()) > 0


@pytest.mark.parametrize('cfg_key', mt5_cfg_reg.list_keys())
def test_mt5_model_and_inference(cfg_key, device): 
    # since MT5Model, MT5Inference simply inherits the T5Model, T5Inference, 
    # we just want to make sure the model can be properly loaded, and leave 
    # the correctness tests to test_model_t5.py
    with device: 
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


def test_mt5_get_pretrained(device): 
    with tempfile.TemporaryDirectory() as root, device: 
        cfg, tokenizer, backbone_params_path, _ = get_pretrained_mt5('google_mt5_small')
        # we exclude <extra_id>s in the comparison below by avoiding len(tokenizer.vocab)
        assert cfg.MODEL.vocab_size >= len(tokenizer._sp_model)
        mt5_model = MT5Model.from_cfg(cfg)
        mt5_model.load_parameters(backbone_params_path)
        mt5_model.hybridize()
        mt5_inference_model = MT5Inference(mt5_model)
        mt5_inference_model.hybridize()
