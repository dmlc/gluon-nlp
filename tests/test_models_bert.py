import pytest
import numpy as np
import os
from numpy.testing import assert_allclose
import mxnet as mx
import tempfile
from gluonnlp.models.bert import BertModel, BertForMLM, BertForPretrain,\
    list_pretrained_bert, get_pretrained_bert
mx.npx.set_np()


def test_list_pretrained_bert():
    assert len(list_pretrained_bert()) > 0


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

@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_bert())
def test_bert(model_name):
    # test from pretrained (fp32, fp16 only on GPU)
    ctx = mx.gpu()
    assert len(list_pretrained_bert()) > 0
    with tempfile.TemporaryDirectory() as root:
        cfg, tokenizer, backbone_params_path, mlm_params_path =\
            get_pretrained_bert(model_name, root=root)
        assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
        bert_model = BertModel.from_cfg(cfg, dtype='float32')
        bert_model.load_parameters(backbone_params_path, ctx=ctx)
        bert_model_fp16 = BertModel.from_cfg(cfg, dtype='float16')
        bert_model_fp16.load_parameters(backbone_params_path, cast_dtype=True, ctx=ctx)

    # test forward fp32 and fp16
    batch_size = 3
    seq_length = 32
    vocab_size = len(tokenizer.vocab)
    input_ids = mx.np.array(
                    np.random.randint(2, vocab_size, (batch_size, seq_length)),
                    dtype=np.int32, ctx=ctx)
    token_types = mx.np.zeros((batch_size, seq_length), np.int32, ctx=ctx)
    valid_length = mx.np.array(
                       np.random.randint(seq_length // 2, seq_length, (batch_size,)),
                       dtype=np.int32, ctx=ctx)
    out_fp32 = bert_model(input_ids, token_types, valid_length)
    mx.npx.waitall()
    # fp16
    os.environ['MXNET_SAFE_ACCUMULATION'] = '1'
    bert_model_fp16.cast('float16')
    bert_model_fp16.hybridize()
    out_fp16 = bert_model_fp16(input_ids, token_types, valid_length)
    mx.npx.waitall()
    os.environ['MXNET_SAFE_ACCUMULATION'] = '0'
    for i, output in enumerate(out_fp32):
        assert_allclose(out_fp32[i].asnumpy(), out_fp16[i].asnumpy(), 1E-1, 1E-1)
