import pytest
import numpy as np
import mxnet as mx
import tempfile
from gluonnlp.models.xlmr import XLMRModel, \
    list_pretrained_xlmr, get_pretrained_xlmr
from gluonnlp.loss import LabelSmoothCrossEntropyLoss

mx.npx.set_np()


def test_list_pretrained_xlmr():
    assert len(list_pretrained_xlmr()) > 0


@pytest.mark.slow
@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_xlmr())
def test_xlmr(model_name, ctx):
    # test from pretrained
    assert len(list_pretrained_xlmr()) > 0
    with ctx:
        with tempfile.TemporaryDirectory() as root:
            cfg, tokenizer, params_path, mlm_params_path =\
                get_pretrained_xlmr(model_name, load_backbone=True, load_mlm=False, root=root)
            assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
            # test backbone
            xlmr_model = XLMRModel.from_cfg(cfg)
            xlmr_model.load_parameters(params_path)
            # pass the mlm model

        # test forward
        batch_size = 1
        seq_length = 4
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
        contextual_embeddings, pooled_out = xlmr_model(input_ids, valid_length)
        mx.npx.waitall()
        # test backward
        label_smooth_loss = LabelSmoothCrossEntropyLoss(num_labels=vocab_size)
        with mx.autograd.record():
            contextual_embeddings, pooled_out = xlmr_model(input_ids, valid_length)
            loss = label_smooth_loss(contextual_embeddings, input_ids)
            loss.backward()
        mx.npx.waitall()
