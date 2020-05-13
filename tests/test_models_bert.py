import pytest
import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
from gluonnlp.models.bert import BertModel, BertForMLM, BertForPretrain,\
    list_pretrained_bert, get_pretrained_bert
mx.npx.set_np()


def test_list_pretrained(tmp_path):
    assert len(list_pretrained_bert()) > 0


@pytest.mark.remote_required
@pytest.mark.parametrize("model_name", list_pretrained_bert())
def test_bert_get_pretrained(tmp_path, model_name):
    cfg, tokenizer, backbone_params_path, mlm_params_path =\
        get_pretrained_bert(model_name, root=str(tmp_path))
    bert_model = BertModel.from_cfg(cfg)
    bert_model.load_parameters(backbone_params_path)
    bert_mlm_model = BertForMLM(cfg)
    if mlm_params_path is not None:
        bert_mlm_model.load_parameters(mlm_params_path)
    bert_mlm_model = BertForMLM(cfg)
    bert_mlm_model.backbone_model.load_parameters(backbone_params_path)
