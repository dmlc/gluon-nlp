import pytest
from numpy.testing import assert_allclose
import mxnet as mx
import tempfile
from gluonnlp.models.bert import BertModel, BertForMLM, BertForPretrain,\
    list_pretrained_bert, get_pretrained_bert
mx.npx.set_np()


def test_list_pretrained_bert():
    assert len(list_pretrained_bert()) > 0


@pytest.mark.parametrize('compute_layout', ['auto', 'NT', 'TN'])
def test_bert_small_cfg(compute_layout, ctx):
    with ctx:
        cfg = BertModel.get_cfg()
        cfg.defrost()
        cfg.MODEL.vocab_size = 100
        cfg.MODEL.units = 12 * 4
        cfg.MODEL.hidden_size = 64
        cfg.MODEL.num_layers = 2
        cfg.MODEL.num_heads = 2
        cfg.MODEL.compute_layout = compute_layout
        cfg.freeze()

        # Generate TN layout
        cfg_tn = cfg.clone()
        cfg_tn.defrost()
        cfg_tn.MODEL.layout = 'TN'
        cfg_tn.freeze()

        # Sample data
        batch_size = 4
        sequence_length = 8
        num_mask = 3
        inputs = mx.np.random.randint(0, 10, (batch_size, sequence_length))
        token_types = mx.np.random.randint(0, 2, (batch_size, sequence_length))
        valid_length = mx.np.random.randint(3, sequence_length, (batch_size,))
        masked_positions = mx.np.random.randint(0, 3, (batch_size, num_mask))

        # Test for BertModel
        bert_model = BertModel.from_cfg(cfg)
        bert_model.initialize()
        bert_model.hybridize()
        contextual_embedding, pooled_out = bert_model(inputs, token_types, valid_length)
        bert_model_tn = BertModel.from_cfg(cfg_tn)
        bert_model_tn.share_parameters(bert_model.collect_params())
        bert_model_tn.hybridize()
        contextual_embedding_tn, pooled_out_tn = bert_model_tn(inputs.T, token_types.T, valid_length)
        assert_allclose(contextual_embedding.asnumpy(),
                        mx.np.swapaxes(contextual_embedding_tn, 0, 1).asnumpy(),
                        1E-4, 1E-4)
        assert_allclose(pooled_out.asnumpy(), pooled_out_tn.asnumpy(), 1E-4, 1E-4)

        # Test for BertForMLM
        bert_mlm_model = BertForMLM(cfg)
        bert_mlm_model.initialize()
        bert_mlm_model.hybridize()
        contextual_embedding, pooled_out, mlm_score = bert_mlm_model(inputs, token_types,
                                                                     valid_length, masked_positions)
        bert_mlm_model_tn = BertForMLM(cfg_tn)
        bert_mlm_model_tn.share_parameters(bert_mlm_model.collect_params())
        bert_mlm_model_tn.hybridize()
        contextual_embedding_tn, pooled_out_tn, mlm_score_tn =\
            bert_mlm_model_tn(inputs.T, token_types.T, valid_length, masked_positions)
        assert_allclose(contextual_embedding.asnumpy(),
                        mx.np.swapaxes(contextual_embedding_tn, 0, 1).asnumpy(),
                        1E-4, 1E-4)
        assert_allclose(pooled_out.asnumpy(), pooled_out_tn.asnumpy(), 1E-4, 1E-4)
        assert_allclose(mlm_score.asnumpy(), mlm_score_tn.asnumpy(), 1E-4, 1E-4)

        # Test for BertForPretrain
        bert_pretrain_model = BertForPretrain(cfg)
        bert_pretrain_model.initialize()
        bert_pretrain_model.hybridize()
        contextual_embedding, pooled_out, nsp_score, mlm_scores =\
            bert_pretrain_model(inputs, token_types, valid_length, masked_positions)
        bert_pretrain_model_tn = BertForPretrain(cfg_tn)
        bert_pretrain_model_tn.share_parameters(bert_pretrain_model.collect_params())
        bert_pretrain_model_tn.hybridize()
        contextual_embedding_tn, pooled_out_tn, nsp_score_tn, mlm_scores_tn = \
            bert_pretrain_model_tn(inputs.T, token_types.T, valid_length, masked_positions)
        assert_allclose(contextual_embedding.asnumpy(),
                        mx.np.swapaxes(contextual_embedding_tn, 0, 1).asnumpy(),
                        1E-4, 1E-4)
        assert_allclose(pooled_out.asnumpy(), pooled_out_tn.asnumpy(), 1E-4, 1E-4)
        assert_allclose(nsp_score.asnumpy(), nsp_score_tn.asnumpy(), 1E-4, 1E-4)
        assert_allclose(mlm_score.asnumpy(), mlm_score_tn.asnumpy(), 1E-4, 1E-4)


@pytest.mark.slow
@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_bert())
def test_bert_get_pretrained(model_name, ctx):
    assert len(list_pretrained_bert()) > 0
    with tempfile.TemporaryDirectory() as root, ctx:
        cfg, tokenizer, backbone_params_path, mlm_params_path =\
            get_pretrained_bert(model_name, load_backbone=True, load_mlm=True, root=root)
        assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
        bert_model = BertModel.from_cfg(cfg)
        bert_model.load_parameters(backbone_params_path)
        bert_mlm_model = BertForMLM(cfg)
        if mlm_params_path is not None:
            bert_mlm_model.load_parameters(mlm_params_path)
        bert_mlm_model = BertForMLM(cfg)
        bert_mlm_model.backbone_model.load_parameters(backbone_params_path)
