import pytest
import numpy as np
import mxnet as mx
import tempfile
from gluonnlp.models.bart import BartModel, \
    list_pretrained_bart, get_pretrained_bart
from gluonnlp.loss import LabelSmoothCrossEntropyLoss

mx.npx.set_np()


def test_list_pretrained_bart():
    assert len(list_pretrained_bart()) > 0


@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_bart())
def test_bart(model_name):
    # test from pretrained
    assert len(list_pretrained_bart()) > 0
    with tempfile.TemporaryDirectory() as root:
        cfg, tokenizer, params_path, _ =\
            get_pretrained_bart(model_name, load_backbone=True, root=root)
        assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
        # test backbone
        bart_model = BartModel.from_cfg(cfg)
        bart_model.load_parameters(params_path)
        # test mlm model
        bart_model_with_pooler = BartModel.from_cfg(
            cfg, use_pooler=True, classifier_activation=False)
        bart_model_with_pooler.load_parameters(params_path)

    # test forward
    batch_size = 3
    seq_length = 32
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
    contextual_embeddings, pooled_out = bart_model_with_pooler(
        input_ids, valid_length, input_ids, valid_length)
    mx.npx.waitall()
    # test backward
    label_smooth_loss = LabelSmoothCrossEntropyLoss(num_labels=vocab_size)
    with mx.autograd.record():
        contextual_embeddings, pooled_out = bart_model_with_pooler(
            input_ids, valid_length, input_ids, valid_length)
        loss = label_smooth_loss(contextual_embeddings, input_ids)
        loss.backward()
    mx.npx.waitall()
