import pytest
import torch as th
from gluonnlp.torch.attention_cell import gen_mem_attn_mask, gen_self_attn_mask
from gluonnlp.torch.models.bert import BertModel, BertForMLM, BertForPretrain, init_weights
from gluonnlp.torch.utils import share_parameters
from numpy.testing import assert_allclose


@pytest.mark.parametrize('compute_layout', ['auto', 'NT', 'TN'])
def test_bert_small(compute_layout):
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
    inputs = th.randint(0, 10, (batch_size, sequence_length))
    token_types = th.randint(0, 2, (batch_size, sequence_length))
    valid_length = th.randint(3, sequence_length, (batch_size, ))
    masked_positions = th.randint(0, 3, (batch_size, num_mask))

    # Test for BertModel
    bert_model = BertModel.from_cfg(cfg)
    bert_model.apply(init_weights)
    bert_model.eval()
    contextual_embedding, pooled_out = bert_model(inputs, token_types, valid_length)
    bert_model_tn = BertModel.from_cfg(cfg_tn)
    share_parameters(bert_model, bert_model_tn)
    bert_model_tn.eval()
    contextual_embedding_tn, pooled_out_tn = bert_model_tn(inputs.T, token_types.T, valid_length)
    assert_allclose(contextual_embedding.detach().numpy(),
                    th.transpose(contextual_embedding_tn, 0, 1).detach().numpy(), 1E-5, 1E-5)
    assert_allclose(pooled_out.detach().numpy(), pooled_out_tn.detach().numpy(), 1E-5, 1E-5)

    # Test for BertForMLM
    bert_mlm_model = BertForMLM(cfg)
    bert_mlm_model.apply(init_weights)
    bert_mlm_model.eval()
    contextual_embedding, pooled_out, mlm_score = bert_mlm_model(inputs, token_types, valid_length,
                                                                 masked_positions)
    bert_mlm_model_tn = BertForMLM(cfg_tn)
    bert_mlm_model_tn.apply(init_weights)
    bert_mlm_model_tn.eval()
    share_parameters(bert_mlm_model, bert_mlm_model_tn)
    contextual_embedding_tn, pooled_out_tn, mlm_score_tn =\
        bert_mlm_model_tn(inputs.T, token_types.T, valid_length, masked_positions)
    assert_allclose(contextual_embedding.detach().numpy(),
                    th.transpose(contextual_embedding_tn, 0, 1).detach().numpy(), 1E-5, 1E-5)
    assert_allclose(pooled_out.detach().numpy(), pooled_out_tn.detach().numpy(), 1E-5, 1E-5)
    assert_allclose(mlm_score.detach().numpy(), mlm_score_tn.detach().numpy(), 1E-5, 1E-5)

    # Test for BertForPretrain
    bert_pretrain_model = BertForPretrain(cfg)
    bert_pretrain_model.apply(init_weights)
    bert_pretrain_model.eval()
    contextual_embedding, pooled_out, nsp_score, mlm_score =\
        bert_pretrain_model(inputs, token_types, valid_length, masked_positions)
    bert_pretrain_model_tn = BertForPretrain(cfg_tn)
    share_parameters(bert_pretrain_model, bert_pretrain_model_tn)
    bert_pretrain_model_tn.eval()
    contextual_embedding_tn, pooled_out_tn, nsp_score_tn, mlm_score_tn = \
        bert_pretrain_model_tn(inputs.T, token_types.T, valid_length, masked_positions)
    assert_allclose(contextual_embedding.detach().numpy(),
                    th.transpose(contextual_embedding_tn, 0, 1).detach().numpy(), 1E-5, 1E-5)
    assert_allclose(pooled_out.detach().numpy(), pooled_out_tn.detach().numpy(), 1E-5, 1E-5)
    assert_allclose(nsp_score.detach().numpy(), nsp_score_tn.detach().numpy(), 1E-5, 1E-5)
    assert_allclose(mlm_score.detach().numpy(), mlm_score_tn.detach().numpy(), 1E-5, 1E-5)
