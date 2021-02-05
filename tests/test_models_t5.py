import pytest
import tempfile

import mxnet as mx
from mxnet import np, npx
from mxnet.gluon import nn, HybridBlock
from gluonnlp.models.t5 import (
    T5Model, T5Inference, t5_cfg_reg, list_pretrained_t5, get_pretrained_t5
)
from gluonnlp.utils.testing import verify_nmt_model, verify_nmt_inference

npx.set_np()


def test_list_pretrained_t5(): 
    assert len(list_pretrained_t5()) > 0


@pytest.mark.parametrize('cfg_key', t5_cfg_reg.list_keys())
@pytest.mark.parametrize('activation', ['relu', 'gated-gelu'])
def test_t5_model(cfg_key, activation, ctx): 
    with ctx: 
        cfg = T5Model.get_cfg(cfg_key)
        cfg.defrost()
        cfg.MODEL.vocab_size = 256
        cfg.MODEL.d_model = 128
        cfg.MODEL.d_ff = 512
        cfg.MODEL.num_layers = 2
        cfg.MODEL.num_heads = 4
        cfg.MODEL.activation = activation
        cfg.MODEL.layout = 'NT'
        cfg.freeze()

        cfg_tn = cfg.clone()
        cfg_tn.defrost()
        cfg_tn.MODEL.layout = 'TN'
        cfg_tn.freeze()
        
        # test TN and NT consistency
        t5_model = T5Model.from_cfg(cfg)
        t5_model.initialize()
        t5_model.hybridize()
        t5_model_tn = T5Model.from_cfg(cfg_tn)
        t5_model_tn.share_parameters(t5_model.collect_params())
        t5_model_tn.hybridize()

        batch_size = 8
        src_length = 32
        tgt_length = 18
        src_data = np.random.randint(0, 255, (batch_size, src_length))
        src_valid_length = np.random.randint(src_length // 2, src_length, (batch_size,))
        tgt_data = np.random.randint(0, 255, (batch_size, tgt_length))
        tgt_valid_length = np.random.randint(tgt_length // 4, tgt_length, (batch_size,))

        out = t5_model(src_data, src_valid_length, tgt_data, tgt_valid_length)
        out_tn = t5_model_tn(src_data.T, src_valid_length, tgt_data.T, tgt_valid_length)
        assert np.allclose(np.swapaxes(out, 0, 1), out_tn, 1E-5, 1E-5)

        # test consistency with various target valid length
        for shift in range(1, np.min(tgt_valid_length).item()): 
            for partial_out in [
                t5_model(src_data, src_valid_length, tgt_data[:, :-shift], tgt_valid_length - shift), 
                t5_model(src_data, src_valid_length, tgt_data, tgt_valid_length - shift)
            ]: 
                for i in range(batch_size):
                    vl = tgt_valid_length[i].item() - shift
                    assert np.allclose(partial_out[i, :vl], out[i, :vl], 1E-5, 1E-5)


@pytest.mark.parametrize('layout', ['NT', 'TN'])
@pytest.mark.parametrize('activation', ['relu', 'gated-gelu'])
def test_t5_inference(layout, activation, ctx): 
    with ctx: 
        cfg = T5Model.get_cfg('google_t5_small')
        cfg.defrost()
        cfg.MODEL.layout = layout
        cfg.MODEL.activation = activation
        cfg.freeze()

        model = T5Model.from_cfg(cfg)
        model.initialize()
        model.hybridize()
        
        # while keeping T5Model implementation consistent with Huggingface's, this 
        # temporary class would help the backbone fit into the provided nmt tests. 
        class TempWithHead(HybridBlock): 
            def __init__(self, model): 
                super().__init__()
                self.model = model
                self.layout = model.layout
                self.src_vocab_size = model.vocab_size
                self.tgt_vocab_size = model.vocab_size
                # append a final output layer
                self.output_layer = nn.Dense(
                    units=model.vocab_size, 
                    in_units=model._d_model, 
                    flatten=False, 
                    use_bias=False, 
                    dtype=model._dtype
                )
                self.output_layer.weight = model.input_embedding_layer.weight

            def forward(self, *args, **kwargs): 
                return self.output_layer(self.model(*args, **kwargs))

        backbone = TempWithHead(model)
        backbone.hybridize()
        verify_nmt_model(backbone)

        inference_model = T5Inference(model)
        inference_model.hybridize()
        verify_nmt_inference(train_model=backbone, inference_model=inference_model)


def test_t5_get_pretrained(ctx): 
    with tempfile.TemporaryDirectory() as root, ctx: 
        cfg, tokenizer, backbone_params_path, _ = get_pretrained_t5('google_t5_small')
        assert cfg.MODEL.vocab_size >= len(tokenizer._sp_model)
        t5_model = T5Model.from_cfg(cfg)
        t5_model.load_parameters(backbone_params_path)
        t5_model.hybridize()
        t5_inference_model = T5Inference(t5_model)
        t5_inference_model.hybridize()
