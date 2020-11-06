import numpy as np
import mxnet as mx
import pytest
from numpy.testing import assert_allclose
from gluonnlp.models.transformer import\
    TransformerEncoder, TransformerDecoder, \
    TransformerModel, TransformerNMTInference,\
    transformer_cfg_reg
from gluonnlp.attention_cell import gen_mem_attn_mask, gen_self_attn_mask
from gluonnlp.utils.testing import verify_nmt_model, verify_nmt_inference
from gluonnlp.utils.testing import verify_backbone_fp16


mx.npx.set_np()


@pytest.mark.parametrize('pre_norm', [False, True])
@pytest.mark.parametrize('num_enc_layers', [2, 3])
@pytest.mark.parametrize('num_dec_layers', [2, 3])
def test_transformer_encoder_decoder(pre_norm, num_enc_layers, num_dec_layers):
    batch_size = 8
    src_seq_length = 20
    tgt_seq_length = 15
    units = 32
    enc = TransformerEncoder(units=units, hidden_size=64, num_layers=num_enc_layers, num_heads=4,
                             dropout=0.0, pre_norm=pre_norm)
    dec = TransformerDecoder(units=units, hidden_size=64, num_layers=num_dec_layers, num_heads=4,
                             dropout=0.0, pre_norm=pre_norm)
    enc.hybridize()
    # disabled due to two different signatures calling attention_cell in this test
    # dec.hybridize()
    enc.initialize()
    dec.initialize()
    src_data = mx.np.random.normal(0, 1, (batch_size, src_seq_length, units))
    src_valid_length = mx.np.random.randint(1, src_seq_length, (batch_size,))
    dst_data = mx.np.random.normal(0, 1, (batch_size, tgt_seq_length, units))
    dst_valid_length = mx.np.random.randint(5, tgt_seq_length, (batch_size,))
    encoded_mem = enc(src_data, src_valid_length)
    full_decode_out = dec(dst_data, dst_valid_length, encoded_mem, src_valid_length)

    # Test for the TN layout
    enc_tn = TransformerEncoder(units=units, hidden_size=64, num_layers=num_enc_layers, num_heads=4,
                                dropout=0.0, pre_norm=pre_norm, layout='TN')
    enc_tn.share_parameters(enc.collect_params())
    dec_tn = TransformerDecoder(units=units, hidden_size=64, num_layers=num_dec_layers, num_heads=4,
                                dropout=0.0, pre_norm=pre_norm, layout='TN')
    dec_tn.share_parameters(dec.collect_params())
    enc_tn.hybridize()
    dec_tn.hybridize()
    encoded_mem_tn = enc_tn(mx.np.swapaxes(src_data, 0, 1), src_valid_length)
    full_decode_out_tn = dec_tn(mx.np.swapaxes(dst_data, 0, 1), dst_valid_length,
                                encoded_mem_tn, src_valid_length)
    assert_allclose(encoded_mem_tn.asnumpy(),
                    mx.np.swapaxes(encoded_mem, 0, 1).asnumpy(), 1E-5, 1E-5)
    assert_allclose(full_decode_out_tn.asnumpy(),
                    mx.np.swapaxes(full_decode_out, 0, 1).asnumpy(), 1E-5, 1E-5)

    # Test the consistency via shifting the data and the valid_length
    for i in range(1, dst_valid_length.asnumpy().min()):
        for partial_decode_out in [dec(dst_data[:, :(-i), :],
                                       dst_valid_length - i, encoded_mem, src_valid_length),
                                   dec(dst_data, dst_valid_length - i,
                                       encoded_mem, src_valid_length)]:
            for b in range(batch_size):
                vl = dst_valid_length.asnumpy()[b] - i
                assert_allclose(partial_decode_out.asnumpy()[b, :vl, :],
                                full_decode_out.asnumpy()[b, :vl, :], 1E-5, 1E-5)
    # Test the decoder layer
    self_causal_mask = gen_self_attn_mask(dst_data, dst_valid_length, attn_type='causal')
    mem_attn_mask = gen_mem_attn_mask(encoded_mem, src_valid_length, dst_data, dst_valid_length)
    enc_mem_attn_mask = gen_mem_attn_mask(encoded_mem, src_valid_length, dst_data[:, 0:1, :],
                                          None)
    print(enc_mem_attn_mask)
    h_out = dec.layers[0](dst_data, encoded_mem, self_causal_mask, mem_attn_mask)
    states = dec.layers[0].init_states(batch_size, h_out.ctx, h_out.dtype)
    h_out_from_incremental = []
    for i in range(tgt_seq_length):
        ele_h_out, states = dec.layers[0].incremental_decode(dst_data[:, i, :], states,
                                                             encoded_mem, src_valid_length,
                                                             enc_mem_attn_mask)
        h_out_from_incremental.append(ele_h_out)
    h_out_from_incremental = mx.np.stack(h_out_from_incremental, axis=1)

    for i in range(batch_size):
        val_length = dst_valid_length[i].asnumpy()
        assert_allclose(h_out_from_incremental[i, :val_length, :].asnumpy(),
                        h_out[i, :val_length, :].asnumpy(), 1E-5, 1E-5)
    # Test for the full decoder
    states = dec.init_states(batch_size, src_data.ctx, src_data.dtype)
    final_out_from_incremental = []
    for i in range(tgt_seq_length):
        ele_final_out, states = dec.incremental_decode(dst_data[:, i, :],
                                                       states, encoded_mem, src_valid_length)
        final_out_from_incremental.append(ele_final_out)
    final_out_from_incremental = mx.np.stack(final_out_from_incremental, axis=1)
    for i in range(batch_size):
        val_length = dst_valid_length[i].asnumpy()
        assert_allclose(final_out_from_incremental[i, :val_length, :].asnumpy(),
                        full_decode_out[i, :val_length, :].asnumpy(), 1E-5, 1E-5)


@pytest.mark.parametrize('train_hybridize,inference_hybridize',
                         [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize('enc_pre_norm,dec_pre_norm',
                         [(False, False), (True, True)])
@pytest.mark.parametrize('enc_num_layers,dec_num_layers,enc_units,dec_units',
                         [(2, 2, 24, 24),
                          (2, 3, 16, 24)])
@pytest.mark.parametrize('enc_recurrent', [False, True])
@pytest.mark.parametrize('dec_recurrent', [False, True])
@pytest.mark.parametrize('tie_weights,layout', [(False, 'NT'), (True, 'NT'), (True, 'TN')])
def test_transformer_nmt_model(train_hybridize, inference_hybridize,
                               enc_pre_norm, dec_pre_norm,
                               enc_units, dec_units,
                               enc_num_layers, dec_num_layers,
                               enc_recurrent, dec_recurrent, tie_weights,
                               layout):
    src_seq_length = 20
    tgt_seq_length = 15
    src_vocab_size = 32
    tgt_vocab_size = 32
    if enc_units != dec_units:
        shared_embed = False
    else:
        shared_embed = True
    model = TransformerModel(src_vocab_size=src_vocab_size,
                             tgt_vocab_size=tgt_vocab_size,
                             max_src_length=src_seq_length,
                             max_tgt_length=tgt_seq_length,
                             enc_units=enc_units,
                             enc_hidden_size=64,
                             enc_num_heads=4,
                             enc_num_layers=enc_num_layers,
                             enc_pre_norm=enc_pre_norm,
                             enc_recurrent=enc_recurrent,
                             dec_units=dec_units,
                             dec_hidden_size=64,
                             dec_num_heads=4,
                             dec_num_layers=dec_num_layers,
                             dec_pre_norm=dec_pre_norm,
                             dec_recurrent=dec_recurrent,
                             shared_embed=shared_embed,
                             tie_weights=tie_weights,
                             dropout=0.0,
                             layout=layout)
    inference_model = TransformerNMTInference(model=model)
    model.initialize()
    if train_hybridize:
        model.hybridize()
    verify_nmt_model(model)
    if inference_hybridize:
        inference_model.hybridize()
    verify_nmt_inference(train_model=model, inference_model=inference_model)


def test_transformer_cfg_registry():
    assert len(transformer_cfg_reg.list_keys()) > 0


@pytest.mark.parametrize('cfg_key', transformer_cfg_reg.list_keys())
def test_transformer_cfg(cfg_key):
    cfg = TransformerModel.get_cfg(cfg_key)
    cfg.defrost()
    cfg.MODEL.src_vocab_size = 32
    cfg.MODEL.tgt_vocab_size = 32
    cfg.freeze()
    model = TransformerModel.from_cfg(cfg)
    model.initialize()
    model.hybridize()
    cfg.defrost()
    cfg.MODEL.layout = 'TN'
    cfg.freeze()
    model_tn = TransformerModel.from_cfg(cfg)
    model_tn.share_parameters(model.collect_params())
    model_tn.hybridize()
    mx.npx.waitall()


@pytest.mark.parametrize('enc_pre_norm,dec_pre_norm',
                         [(False, False), (True, True)])
@pytest.mark.parametrize('enc_num_layers,dec_num_layers,enc_units,dec_units',
                         [(2, 2, 24, 24),
                          (2, 3, 16, 16)])
@pytest.mark.parametrize('enc_recurrent', [False, True])
@pytest.mark.parametrize('dec_recurrent', [False, True])
@pytest.mark.parametrize('tie_weights,layout', [(False, 'NT'), (True, 'NT'), (True, 'TN')])
def test_transformer_fp16_amp(enc_pre_norm, dec_pre_norm,
                              enc_units, dec_units,
                              enc_num_layers, dec_num_layers,
                              enc_recurrent, dec_recurrent, tie_weights,
                              layout, ctx):
    if ctx.device_type != 'gpu':
        pytest.skip('Only test amp when running on GPU.')
    # Generate configuration for testing
    cfg = TransformerModel.get_cfg()
    cfg.defrost()
    cfg.MODEL.src_vocab_size = 32
    cfg.MODEL.tgt_vocab_size = 32
    cfg.MODEL.max_src_length = 20
    cfg.MODEL.max_tgt_length = 15
    cfg.MODEL.tie_weights = tie_weights
    cfg.MODEL.layout = layout

    # Encoder config
    cfg.MODEL.ENCODER.pre_norm = enc_pre_norm
    cfg.MODEL.ENCODER.units = enc_units
    cfg.MODEL.ENCODER.num_layers = enc_num_layers
    cfg.MODEL.ENCODER.recurrent = enc_recurrent

    # Decoder config
    cfg.MODEL.DECODER.pre_norm = dec_pre_norm
    cfg.MODEL.DECODER.units = dec_units
    cfg.MODEL.DECODER.num_layers = dec_num_layers
    cfg.MODEL.DECODER.recurrent = dec_recurrent
    cfg.freeze()

    batch_size = 4
    seq_length = 16
    with ctx:
        if layout == 'NT':
            src_data = mx.np.random.randint(0, cfg.MODEL.src_vocab_size,
                                            (batch_size, seq_length), dtype=np.int32)
            src_valid_length = mx.np.random.randint(seq_length // 2, seq_length,
                                                    (batch_size,), dtype=np.int32)
            tgt_data = mx.np.random.randint(0, cfg.MODEL.tgt_vocab_size,
                                            (batch_size, seq_length), dtype=np.int32)
            tgt_valid_length = mx.np.random.randint(seq_length // 2, seq_length,
                                                    (batch_size,), dtype=np.int32)
        elif layout == 'TN':
            src_data = mx.np.random.randint(0, cfg.MODEL.src_vocab_size,
                                            (seq_length, batch_size), dtype=np.int32)
            src_valid_length = mx.np.random.randint(seq_length // 2, seq_length,
                                                    (batch_size,), dtype=np.int32)
            tgt_data = mx.np.random.randint(0, cfg.MODEL.tgt_vocab_size,
                                            (seq_length, batch_size), dtype=np.int32)
            tgt_valid_length = mx.np.random.randint(seq_length // 2, seq_length,
                                                    (batch_size,), dtype=np.int32)
        else:
            raise NotImplementedError
        verify_backbone_fp16(TransformerModel, cfg, ctx,
                             inputs=[src_data, src_valid_length, tgt_data, tgt_valid_length])
