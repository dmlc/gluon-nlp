import pytest
import mxnet as mx
from mxnet import np, npx
from gluonnlp.models.t5 import (
    T5Model, list_pretrained_t5, get_pretrained_t5
)


npx.set_np()


def test_list_pretrained_t5(): 
    assert len(list_pretrained_t5()) > 0


@pytest.mark.parametrize('activation', ['relu', 'gated-gelu'])
def test_t5_small_config(activation, ctx): 
    with ctx: 
        cfg = T5Model.get_cfg()
        cfg.defrost()
        cfg.MODEL.vocab_size = 256
        cfg.MODEL.d_model = 128
        cfg.MODEL.d_ff = 512
        cfg.MODEL.num_layers = 2
        cfg.MODEL.num_heads = 4
        cfg.MODEL.layout = 'NT'
        cfg.freeze()

        cfg_tn = cfg.clone()
        cfg_tn.defrost()
        cfg_tn.MODEL.layout = 'TN'
        cfg_tn.freeze()
        
        # T5Model
        t5_model = T5Model.from_cfg(cfg)
        t5_model.initialize()
        t5_model_tn = T5Model.from_cfg(cfg_tn)
        t5_model_tn.share_parameters(t5_model.collect_params())

        batch_size = 4
        src_length = 32
        tgt_length = 16
        src_data = np.random.randint(0, 255, (batch_size, src_length))
        src_valid_length = np.random.randint(2, src_length, (batch_size,))
        tgt_data = np.random.randint(0, 255, (batch_size, tgt_length))
        tgt_valid_length = np.random.randint(5, tgt_length, (batch_size,))

        out = t5_model(src_data, src_valid_length, tgt_data, tgt_valid_length)
        out_tn = t5_model_tn(src_data.T, src_valid_length, tgt_data.T, tgt_valid_length)
        assert np.allclose(np.swapaxes(out, 0, 1), out_tn, 1E-3, 1E-3)

        # T5Inference
        # TODO(yongyi-wu)
