import pytest
import numpy as np
import mxnet as mx
import tempfile
from numpy.testing import assert_allclose
from gluonnlp.models.roberta import RobertaModel, RobertaForMLM, \
    list_pretrained_roberta, get_pretrained_roberta
from gluonnlp.loss import LabelSmoothCrossEntropyLoss

mx.npx.set_np()


def test_list_pretrained_roberta():
    assert len(list_pretrained_roberta()) > 0


@pytest.mark.parametrize('compute_layout', ['auto', 'TN', 'NT'])
def test_robert_small_config(compute_layout):
    cfg = RobertaModel.get_cfg()
    cfg.defrost()
    cfg.MODEL.vocab_size = 1000
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
    valid_length = mx.np.random.randint(3, sequence_length, (batch_size,))
    masked_positions = mx.np.random.randint(0, 3, (batch_size, num_mask))

    roberta_model = RobertaModel.from_cfg(cfg)
    roberta_model.initialize()
    roberta_model.hybridize()
    contextual_embeddings, pooled_out = roberta_model(inputs, valid_length)
    roberta_model_tn = RobertaModel.from_cfg(cfg_tn)
    roberta_model_tn.share_parameters(roberta_model.collect_params())
    roberta_model_tn.hybridize()
    contextual_embeddings_tn, pooled_out_tn = roberta_model_tn(inputs.T, valid_length)
    assert_allclose(np.swapaxes(contextual_embeddings_tn.asnumpy(), 0, 1),
                    contextual_embeddings.asnumpy(), 1E-4, 1E-4)
    assert_allclose(pooled_out_tn.asnumpy(), pooled_out.asnumpy(), 1E-4, 1E-4)

    # Test for RobertaForMLM
    roberta_mlm_model = RobertaForMLM(cfg)
    roberta_mlm_model.initialize()
    roberta_mlm_model.hybridize()
    contextual_embedding, pooled_out, mlm_scores = roberta_mlm_model(inputs, valid_length,
                                                                     masked_positions)
    roberta_mlm_model_tn = RobertaForMLM(cfg_tn)
    roberta_mlm_model_tn.share_parameters(roberta_mlm_model.collect_params())
    roberta_mlm_model_tn.hybridize()
    contextual_embedding_tn, pooled_out_tn, mlm_scores_tn =\
        roberta_mlm_model_tn(inputs.T, valid_length.T, masked_positions)
    assert_allclose(np.swapaxes(contextual_embedding_tn.asnumpy(), 0, 1),
                    contextual_embedding.asnumpy(), 1E-4, 1E-4)
    assert_allclose(pooled_out_tn.asnumpy(), pooled_out.asnumpy(), 1E-4, 1E-4)
    assert_allclose(mlm_scores_tn.asnumpy(), mlm_scores.asnumpy(), 1E-4, 1E-4)


@pytest.mark.slow
@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_roberta())
def test_roberta(model_name):
    # test from pretrained
    assert len(list_pretrained_roberta()) > 0
    with tempfile.TemporaryDirectory() as root:
        cfg, tokenizer, params_path, mlm_params_path =\
            get_pretrained_roberta(model_name, load_backbone=True, load_mlm=True, root=root)
        assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
        # test backbone
        roberta_model = RobertaModel.from_cfg(cfg)
        roberta_model.load_parameters(params_path)
        # test mlm model
        roberta_mlm_model = RobertaForMLM(cfg)
        if mlm_params_path is not None:
            roberta_mlm_model.load_parameters(mlm_params_path)
        roberta_mlm_model = RobertaForMLM(cfg)
        roberta_mlm_model.backbone_model.load_parameters(params_path)

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
