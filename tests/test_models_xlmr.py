import pytest
import numpy as np
import mxnet as mx
import tempfile
from gluonnlp.models.xlmr import XLMRModel,\
    list_pretrained_xlmr, get_pretrained_xlmr
from gluonnlp.loss import LabelSmoothCrossEntropyLoss

mx.npx.set_np()


def test_list_pretrained_xlmr():
    assert len(list_pretrained_xlmr()) > 0


@pytest.mark.remote_required
def test_xlmr():
    # test from pretrained
    assert len(list_pretrained_xlmr()) > 0
    for model_name in list_pretrained_xlmr():
        with tempfile.TemporaryDirectory() as root:
            cfg, tokenizer, params_path =\
                get_pretrained_xlmr(model_name, root=root)
            assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
            xlmr_model = XLMRModel.from_cfg(cfg)
            xlmr_model.load_parameters(params_path)
        # test forward
        batch_size = 2
        seq_length = 16
        vocab_size = len(tokenizer.vocab)
        input_ids = mx.np.array(
            np.random.randint(
                2,
                vocab_size,
                (batch_size, seq_length)
            ),
            dtype=np.int32
        )
        valid_length = mx.np.array(
            np.random.randint(
                seq_length // 2,
                seq_length,
                (batch_size,)
            ),
            dtype=np.int32
        )
        x = xlmr_model(input_ids, valid_length)
        mx.npx.waitall()
        # test backward
        label_smooth_loss = LabelSmoothCrossEntropyLoss(num_labels=vocab_size)
        with mx.autograd.record():
            x = xlmr_model(input_ids, valid_length)
            loss = label_smooth_loss(x, input_ids)
            loss.backward()
        mx.npx.waitall()
