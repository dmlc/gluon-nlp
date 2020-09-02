import pytest
import numpy as np
import mxnet as mx
import tempfile
from numpy.testing import assert_allclose
from gluonnlp.models.gpt2 import GPT2Model, GPT2ForLM, \
    list_pretrained_gpt2, get_pretrained_gpt2
from gluonnlp.loss import LabelSmoothCrossEntropyLoss

mx.npx.set_np()


def test_list_pretrained_gpt2():
    assert len(list_pretrained_gpt2()) > 0


@pytest.mark.parametrize('compute_layout', ['auto', 'TN', 'NT'])
def test_gpt2_small_config(compute_layout, ctx):
    cfg = GPT2Model.get_cfg()
    cfg.defrost()
    cfg.MODEL.vocab_size = 1000
    cfg.MODEL.units = 128
    cfg.MODEL.num_layers = 2
    cfg.MODEL.num_heads = 2
    cfg.MODEL.compute_layout = compute_layout
    cfg.freeze()

    # Generate TN layout
    cfg_tn = cfg.clone()
    cfg_tn.defrost()
    cfg_tn.MODEL.layout = 'TN'
    cfg_tn.freeze()
    
    with ctx:
        batch_size = 4
        sequence_length = 16
        inputs = mx.np.random.randint(0, 1000, (batch_size, sequence_length), ctx=ctx)

        gpt2_model = GPT2Model.from_cfg(cfg)
        gpt2_model.initialize(ctx=ctx)
        gpt2_model.hybridize()
        hiddens, _ = gpt2_model(
            inputs,
            gpt2_model.init_states(batch_size, ctx),
            mx.np.array(0, dtype=np.int32, ctx=ctx)
        )
        gpt2_model_tn = GPT2Model.from_cfg(cfg_tn)
        gpt2_model_tn.share_parameters(gpt2_model.collect_params())
        gpt2_model_tn.hybridize()
        hiddens_tn, _ = gpt2_model_tn(
            inputs.T,
            gpt2_model_tn.init_states(batch_size, ctx),
            mx.np.array(0, dtype=np.int32, ctx=ctx)
        )
        assert_allclose(np.swapaxes(hiddens_tn.asnumpy(), 0, 1),
                        hiddens.asnumpy(), 1E-4, 1E-4)

        # Test for GPT2ForLM
        gpt2_lm_model = GPT2ForLM(cfg)
        gpt2_lm_model.initialize(ctx=ctx)
        gpt2_lm_model.hybridize()
        logits, states = gpt2_lm_model(
            inputs,
            gpt2_lm_model.init_states(batch_size, ctx),
            mx.np.array(0, dtype=np.int32, ctx=ctx)
        )
        gpt2_lm_model_tn = GPT2ForLM(cfg_tn)
        gpt2_lm_model_tn.share_parameters(gpt2_lm_model.collect_params())
        gpt2_lm_model_tn.hybridize()
        logits_tn, states_tn = gpt2_lm_model_tn(
            inputs.T,
            gpt2_lm_model_tn.init_states(batch_size, ctx),
            mx.np.array(0, dtype=np.int32, ctx=ctx)
        )
        assert_allclose(np.swapaxes(logits_tn.asnumpy(), 0, 1),
                        logits.asnumpy(), 1E-4, 1E-4)
        assert_allclose(np.swapaxes(states_tn.asnumpy(), 2, 3),
                        states.asnumpy(), 1E-4, 1E-4)


def test_gpt2_incremental_states(ctx):
    with ctx:
        batch_size = 4
        sequence_length = 5
        inputs = mx.np.random.randint(0, 1000, (batch_size, sequence_length), ctx=ctx)

        cfg = GPT2Model.get_cfg()
        gpt2_model = GPT2Model.from_cfg(cfg)
        gpt2_model.initialize(ctx=ctx)
        gpt2_model.hybridize()

        one_time_hiddens, one_time_states = gpt2_model(
            inputs,
            gpt2_model.init_states(batch_size, ctx),
            mx.np.array(0, dtype=np.int32, ctx=ctx)
        )

        states = gpt2_model.init_states(batch_size, ctx)
        hiddens_l = []
        for i in range(sequence_length):
            hiddens, states = gpt2_model(
                inputs[:, i:i+1],
                states,
                mx.np.array(i, dtype=np.int32, ctx=ctx)
            )
            hiddens_l.append(hiddens)
        hiddens_concat = mx.np.concatenate(hiddens_l, axis=1)
        assert_allclose(one_time_states.asnumpy(),
                        states.asnumpy(), 1E-4, 1E-4)
        assert_allclose(one_time_hiddens.asnumpy(),
                        hiddens_concat.asnumpy(), 1E-4, 1E-4)


@pytest.mark.slow
@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', list_pretrained_gpt2())
def test_gpt2(model_name, ctx):
    # test from pretrained
    assert len(list_pretrained_gpt2()) > 0
    with tempfile.TemporaryDirectory() as root, ctx:
        cfg, tokenizer, params_path, lm_params_path =\
            get_pretrained_gpt2(model_name, load_backbone=True, load_lm=True, root=root)
        assert cfg.MODEL.vocab_size == len(tokenizer.vocab)
        # test backbone
        gpt2_model = GPT2Model.from_cfg(cfg)
        gpt2_model.load_parameters(params_path)
        # test lm model
        gpt2_lm_model = GPT2ForLM(cfg)
        gpt2_lm_model.load_parameters(lm_params_path)

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
            dtype=np.int32,
            ctx=ctx
        )
        logits, _ = gpt2_lm_model(
            input_ids,
            gpt2_lm_model.init_states(batch_size, ctx),
            mx.np.array(0, dtype=np.int32, ctx=ctx)
        )
        mx.npx.waitall()
        # test backward
        label_smooth_loss = LabelSmoothCrossEntropyLoss(num_labels=vocab_size)
        with mx.autograd.record():
            logits, _ = gpt2_lm_model(
                input_ids,
                gpt2_lm_model.init_states(batch_size, ctx),
                mx.np.array(0, dtype=np.int32, ctx=ctx)
            )
            loss = label_smooth_loss(logits, input_ids)
            loss.backward()
        mx.npx.waitall()
