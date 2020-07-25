import pytest
import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
import tempfile
from gluonnlp.models.mobilebert import MobileBertModel, MobileBertForMLM, MobileBertForPretrain,\
    list_pretrained_mobilebert, get_pretrained_mobilebert
mx.npx.set_np()


def test_list_pretrained_mobilebert():
    assert len(list_pretrained_mobilebert()) > 0


@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_mobilebert())
def test_bert_get_pretrained(model_name):
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
