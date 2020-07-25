import pytest
from gluonnlp.models.transformer_xl import TransformerXLForLM
import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose
from gluonnlp.utils.parameter import grad_global_norm
mx.npx.set_np()


@pytest.mark.parametrize('cutoffs,div_val',
                         [([], 1.0), ([10, 50], 2.0)])
@pytest.mark.parametrize('mem_length,query_length',
                         [(20, 20), (10, 6), (6, 10)])
def test_transformer_xl_for_lm(cutoffs, div_val, mem_length, query_length):
    vocab_size = 100
    cfg = TransformerXLForLM.get_cfg()
    cfg.defrost()
    cfg.MODEL.vocab_size = vocab_size
    cfg.MODEL.embed_units = 48
    cfg.MODEL.units = 32
    cfg.MODEL.hidden_size = 64
    cfg.MODEL.num_layers = 2
    cfg.MODEL.cutoffs = cutoffs
    cfg.MODEL.div_val = div_val
    cfg.MODEL.layout = 'NT'
    cfg.MODEL.dropout = 0.0
    cfg.MODEL.activation_dropout = 0.0
    cfg.MODEL.attention_dropout = 0.0
    cfg.freeze()
    nt_model = TransformerXLForLM(cfg)
    nt_model.initialize()

    tn_cfg = cfg.clone()
    tn_cfg.defrost()
    tn_cfg.MODEL.layout = 'TN'
    tn_model = TransformerXLForLM(tn_cfg)
    tn_model.initialize()
    for name, param in tn_model.collect_params().items():
        param.set_data(nt_model.collect_params().get(name).data())
    assert_allclose(sum(
        mx.np.linalg.norm(param.data()).asnumpy() for param in nt_model.collect_params().values()),
                    sum(mx.np.linalg.norm(param.data()).asnumpy() for param in
                        tn_model.collect_params().values()))
    batch_size = 3
    nt_model.set_mem_length(mem_length)
    tn_model.set_mem_length(mem_length)

    ctx = mx.cpu()

    data = mx.np.random.randint(0, vocab_size, (batch_size, query_length), ctx=ctx, dtype=np.int32)
    target = mx.np.random.randint(0, vocab_size, (batch_size, query_length), ctx=ctx,
                                  dtype=np.int32)

    # Check consistency of layout
    nt_mem_l = nt_model.init_states(batch_size, ctx=ctx)
    for _ in range(8):
        with mx.autograd.record():
            nt_logits, nt_mem_l = nt_model(data, target, nt_mem_l)
            loss = nt_logits.sum()
            loss.backward()
    tn_mem_l = tn_model.init_states(batch_size, ctx=ctx)
    for _ in range(8):
        with mx.autograd.record():
            tn_logits, tn_mem_l = tn_model(data.T, target.T, tn_mem_l)
            loss = tn_logits.sum()
            loss.backward()
    assert_allclose(tn_logits.T.asnumpy(), nt_logits.asnumpy(), 1E-5, 1E-5)
    for name, tn_param in tn_model.collect_params().items():
        nt_param = nt_model.collect_params().get(name)
        if nt_param.grad_req != 'null':
            assert_allclose(nt_param.grad().asnumpy(), tn_param.grad().asnumpy(), 1E-4, 1E-4)

    # Check step_forward consistency
    mem_l = nt_model.init_states(batch_size, ctx=ctx)
    sel_logits, new_mem_l = nt_model(data, target, mem_l)
    ele_sel_logits_l = []
    step_new_mem_l = mem_l
    for i in range(query_length):
        step_logits, step_new_mem_l = nt_model.step_forward(data[:, i], step_new_mem_l)
        ele_sel_logits_l.append(step_logits[mx.np.arange(batch_size), target[:, i]])
    sel_logits_from_step = mx.np.stack(ele_sel_logits_l, axis=-1)
    assert_allclose(sel_logits_from_step.asnumpy(), sel_logits.asnumpy(), 1E-4, 1E-4)
    for lhs, rhs in zip(step_new_mem_l, new_mem_l):
        assert_allclose(lhs.asnumpy(), rhs.asnumpy(), 1E-4, 1E-4)
