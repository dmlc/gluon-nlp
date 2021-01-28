import pytest
import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
import tempfile
from gluonnlp.models.mobilebert import MobileBertModel, MobileBertForMLM, MobileBertForPretrain,\
    list_pretrained_mobilebert, get_pretrained_mobilebert
from gluonnlp.utils.testing import verify_backbone_fp16
mx.npx.set_np()


def test_list_pretrained_mobilebert():
    assert len(list_pretrained_mobilebert()) > 0


@pytest.mark.parametrize('compute_layout', ['auto', 'TN', 'NT'])
def test_mobilebert_model_small_cfg(compute_layout, ctx):
    with ctx:
        cfg = MobileBertModel.get_cfg()
        cfg.defrost()
        cfg.MODEL.vocab_size = 100
        cfg.MODEL.num_layers = 2
        cfg.MODEL.hidden_size = 128
        cfg.MODEL.num_heads = 2
        cfg.MODEL.compute_layout = compute_layout
        cfg.freeze()

        # Generate TN layout
        cfg_tn = cfg.clone()
        cfg_tn.defrost()
        cfg_tn.MODEL.layout = 'TN'
        cfg_tn.freeze()

        batch_size = 4
        sequence_length = 16
        num_mask = 3
        inputs = mx.np.random.randint(0, 10, (batch_size, sequence_length))
        token_types = mx.np.random.randint(0, 2, (batch_size, sequence_length))
        valid_length = mx.np.random.randint(3, sequence_length, (batch_size,))
        masked_positions = mx.np.random.randint(0, 3, (batch_size, num_mask))

        mobile_bert_model = MobileBertModel.from_cfg(cfg)
        mobile_bert_model.initialize()
        mobile_bert_model.hybridize()
        mobile_bert_model_tn = MobileBertModel.from_cfg(cfg_tn)
        mobile_bert_model_tn.share_parameters(mobile_bert_model.collect_params())
        mobile_bert_model_tn.hybridize()
        contextual_embedding, pooled_out = mobile_bert_model(inputs, token_types, valid_length)
        contextual_embedding_tn, pooled_out_tn = mobile_bert_model_tn(inputs.T,
                                                                      token_types.T, valid_length)
        assert_allclose(contextual_embedding.asnumpy(),
                        np.swapaxes(contextual_embedding_tn.asnumpy(), 0, 1),
                        1E-3, 1E-3)
        assert_allclose(pooled_out.asnumpy(), pooled_out_tn.asnumpy(), 1E-3, 1E-3)

        # Test for MobileBertForMLM
        mobile_bert_mlm_model = MobileBertForMLM(cfg)
        mobile_bert_mlm_model.initialize()
        mobile_bert_mlm_model.hybridize()
        mobile_bert_mlm_model_tn = MobileBertForMLM(cfg_tn)
        mobile_bert_mlm_model_tn.share_parameters(mobile_bert_mlm_model.collect_params())
        mobile_bert_model_tn.hybridize()
        contextual_embedding, pooled_out, mlm_score = mobile_bert_mlm_model(inputs, token_types,
                                                                             valid_length,
                                                                             masked_positions)
        contextual_embedding_tn, pooled_out_tn, mlm_score_tn =\
            mobile_bert_mlm_model_tn(inputs.T, token_types.T, valid_length, masked_positions)
        assert_allclose(contextual_embedding.asnumpy(),
                        np.swapaxes(contextual_embedding_tn.asnumpy(), 0, 1),
                        1E-3, 1E-3)
        assert_allclose(pooled_out_tn.asnumpy(), pooled_out.asnumpy(), 1E-3, 1E-3)
        assert_allclose(mlm_score_tn.asnumpy(), mlm_score.asnumpy(), 1E-3, 1E-3)

        # Test for MobileBertForPretrain
        mobile_bert_pretrain_model = MobileBertForPretrain(cfg)
        mobile_bert_pretrain_model.initialize()
        mobile_bert_pretrain_model.hybridize()
        mobile_bert_pretrain_model_tn = MobileBertForPretrain(cfg_tn)
        mobile_bert_pretrain_model_tn.share_parameters(mobile_bert_pretrain_model.collect_params())
        mobile_bert_pretrain_model_tn.hybridize()
        contextual_embedding, pooled_out, nsp_score, mlm_score =\
            mobile_bert_pretrain_model(inputs, token_types, valid_length, masked_positions)
        contextual_embedding_tn, pooled_out_tn, nsp_score_tn, mlm_score_tn = \
            mobile_bert_pretrain_model_tn(inputs.T, token_types.T, valid_length, masked_positions)
        assert_allclose(contextual_embedding.asnumpy(),
                        np.swapaxes(contextual_embedding_tn.asnumpy(), 0, 1),
                        1E-3, 1E-3)
        assert_allclose(pooled_out.asnumpy(), pooled_out_tn.asnumpy(), 1E-3, 1E-3)
        assert_allclose(nsp_score.asnumpy(), nsp_score_tn.asnumpy(), 1E-3, 1E-3)
        assert_allclose(mlm_score.asnumpy(), mlm_score_tn.asnumpy(), 1E-3, 1E-3)

        # Test for fp16
        if ctx.device_type == 'gpu':
            pytest.skip('MobileBERT will have nan values in FP16 mode.')
            verify_backbone_fp16(model_cls=MobileBertModel, cfg=cfg, ctx=ctx,
                                 inputs=[inputs, token_types, valid_length])


@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_mobilebert())
def test_mobilebert_get_pretrained(model_name):
    with tempfile.TemporaryDirectory() as root:
        cfg, tokenizer, backbone_params_path, mlm_params_path =\
            get_pretrained_mobilebert(model_name, load_backbone=True, load_mlm=True, root=root)
        assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
        mobilebert_model = MobileBertModel.from_cfg(cfg)
        mobilebert_model.load_parameters(backbone_params_path)
        mobilebert_pretain_model = MobileBertForPretrain(cfg)
        if mlm_params_path is not None:
            mobilebert_pretain_model.load_parameters(mlm_params_path)
        mobilebert_pretain_model = MobileBertForPretrain(cfg)
        mobilebert_pretain_model.backbone_model.load_parameters(backbone_params_path)
