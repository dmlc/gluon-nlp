import pytest
import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
import tempfile
from gluonnlp.models.bert import BertModel, BertForMLM, BertForPretrain,\
    list_pretrained_bert, get_pretrained_bert
mx.npx.set_np()


def test_list_pretrained_bert():
    assert len(list_pretrained_bert()) > 0


def test_bert_small_cfg():
    cfg = BertModel.get_cfg()
    cfg.defrost()
    cfg.MODEL.vocab_size = 100
    cfg.MODEL.units = 12 * 8
    cfg.MODEL.num_layers = 2
    cfg.freeze()
    cfg_tn = cfg.clone()
    cfg_tn.defrost()
    cfg_tn.MODEL.layout = 'TN'
    cfg_tn.freeze()
    batch_size = 4
    sequence_length = 16
    inputs = mx.np.random.randint(0, 10, (batch_size, sequence_length))
    token_types = mx.np.random.randint(0, 2, (batch_size, sequence_length))
    valid_length = mx.np.random.randint(1, sequence_length, (batch_size,))
    bert_model = BertModel.from_cfg(cfg)
    bert_model.initialize()
    bert_model.hybridize()
    bert_model_tn = BertModel.from_cfg(cfg_tn, params=bert_model.collect_params())
    bert_model_tn.hybridize()
    contextual_embedding, pooled_out = bert_model(inputs, token_types, valid_length)
    contextual_embedding_tn, pooled_out_tn = bert_model_tn(inputs.T, token_types.T, valid_length)
    assert_allclose(contextual_embedding.asnumpy(),
                    mx.np.swapaxes(contextual_embedding_tn, 0, 1).asnumpy(),
                    1E-4, 1E-4)
    assert_allclose(pooled_out.asnumpy(), pooled_out_tn.asnumpy(), 1E-4, 1E-4)


@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_bert())
def test_bert_get_pretrained(model_name):
    assert len(list_pretrained_bert()) > 0
    with tempfile.TemporaryDirectory() as root:
        cfg, tokenizer, backbone_params_path, mlm_params_path =\
            get_pretrained_bert(model_name, root=root)
        assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
        bert_model = BertModel.from_cfg(cfg)
        bert_model.load_parameters(backbone_params_path)
        bert_mlm_model = BertForMLM(cfg)
        if mlm_params_path is not None:
            bert_mlm_model.load_parameters(mlm_params_path)
        bert_mlm_model = BertForMLM(cfg)
        bert_mlm_model.backbone_model.load_parameters(backbone_params_path)
