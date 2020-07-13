import pytest
import numpy as np
import mxnet as mx
import tempfile
from gluonnlp.models.roberta import RobertaModel, RobertaForMLM, \
    list_pretrained_roberta, get_pretrained_roberta
from gluonnlp.loss import LabelSmoothCrossEntropyLoss

mx.npx.set_np()


def test_list_pretrained_roberta():
    assert len(list_pretrained_roberta()) > 0


@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_roberta())
def test_roberta(model_name):
    # test from pretrained
    assert len(list_pretrained_roberta()) > 0
    with tempfile.TemporaryDirectory() as root:
        cfg, tokenizer, params_path, mlm_params_path =\
            get_pretrained_roberta(model_name, load_backbone=True, load_mlm=True, root=root)
        assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
        roberta_model = RobertaModel.from_cfg(cfg)
        roberta_model.load_parameters(params_path)

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
    contextual_embeddings, pooled_out = roberta_model(input_ids, valid_length)
    mx.npx.waitall()
    # test backward
    label_smooth_loss = LabelSmoothCrossEntropyLoss(num_labels=vocab_size)
    with mx.autograd.record():
        contextual_embeddings, pooled_out = roberta_model(input_ids, valid_length)
        loss = label_smooth_loss(contextual_embeddings, input_ids)
        loss.backward()
    mx.npx.waitall()

    # test for mlm model
    roberta_mlm_model = RobertaForMLM(cfg)
    if mlm_params_path is not None:
        roberta_mlm_model.load_parameters(mlm_params_path)
    roberta_mlm_model = RobertaForMLM(cfg)
    roberta_mlm_model.backbone_model.load_parameters(params_path)
