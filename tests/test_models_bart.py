import pytest
import mxnet as mx
import tempfile
import numpy as np
import numpy.testing as npt
from gluonnlp.models.bart import BartModel, \
    list_pretrained_bart, get_pretrained_bart, bart_cfg_reg
from gluonnlp.utils.testing import verify_backbone_fp16


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
def test_bart_cfg(cfg_key, ctx):
    cfg = BartModel.get_cfg(cfg_key)
    cfg.defrost()
    cfg.MODEL.vocab_size = 32
    cfg.freeze()

    cfg_tn = cfg.clone()
    cfg_tn.defrost()
    cfg_tn.MODEL.layout = 'TN'
    cfg_tn.freeze()

    batch_size = 4
    src_length = 32
    tgt_length = 16

    with ctx:
        src_data = mx.np.random.randint(0, cfg.MODEL.vocab_size, (batch_size, src_length))
        src_valid_length = mx.np.random.randint(src_length // 2, src_length, (batch_size,))
        tgt_data = mx.np.random.randint(0, cfg.MODEL.vocab_size, (batch_size, tgt_length))
        tgt_valid_length = mx.np.random.randint(tgt_length // 2, tgt_length, (batch_size, ))
        model = BartModel.from_cfg(cfg)
        model.initialize()
        model.hybridize()

        contextual_embedding, pooled_output = model(src_data, src_valid_length,
                                                    tgt_data, tgt_valid_length)
        model_tn = BartModel.from_cfg(cfg_tn)
        model_tn.share_parameters(model.collect_params())
        model_tn.hybridize()
        contextual_embedding_tn, pooled_out_tn = model_tn(src_data.T, src_valid_length,
                                                          tgt_data.T, tgt_valid_length)
        npt.assert_allclose(contextual_embedding.asnumpy(),
                            np.transpose(contextual_embedding_tn.asnumpy(), (1, 0, 2)), 1E-3, 1E-3)
        npt.assert_allclose(pooled_out_tn.asnumpy(), pooled_output.asnumpy(), 1E-3, 1E-3)
        mx.npx.waitall()

        # Verify Float16
        if ctx.device_type == 'gpu':
            verify_backbone_fp16(model_cls=BartModel, cfg=cfg, ctx=ctx,
                                 inputs=[src_data, src_valid_length, tgt_data, tgt_valid_length])
