import mxnet as mx
import pytest
from numpy.testing import assert_allclose
from gluonnlp.models.transformer import\
    TransformerEncoder, TransformerDecoder, \
    TransformerNMTModel, TransformerNMTInference,\
    transformer_nmt_cfg_reg
from gluonnlp.attention_cell import gen_mem_attn_mask, gen_self_attn_mask
from gluonnlp.utils.testing import verify_nmt_model, verify_nmt_inference
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
                             dropout=0.0, pre_norm=pre_norm, prefix='enc_')
    dec = TransformerDecoder(units=units, hidden_size=64, num_layers=num_dec_layers, num_heads=4,
                             dropout=0.0, pre_norm=pre_norm, prefix='dec_')
    enc.hybridize()
    dec.hybridize()
    enc.initialize()
    dec.initialize()
    src_data = mx.np.random.normal(0, 1, (batch_size, src_seq_length, units))
    src_valid_length = mx.np.random.randint(1, src_seq_length, (batch_size,))
    dst_data = mx.np.random.normal(0, 1, (batch_size, tgt_seq_length, units))
    dst_valid_length = mx.np.random.randint(5, tgt_seq_length, (batch_size,))
    encoded_mem = enc(src_data, src_valid_length)
    full_decode_out = dec(dst_data, dst_valid_length, encoded_mem, src_valid_length)

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
    self_causal_mask = gen_self_attn_mask(mx, dst_data, dst_valid_length, attn_type='causal')
    mem_attn_mask = gen_mem_attn_mask(mx, encoded_mem, src_valid_length, dst_data, dst_valid_length)
    enc_mem_attn_mask = gen_mem_attn_mask(mx, encoded_mem, src_valid_length, dst_data[:, 0:1, :],
                                          None)
    h_out = dec.layers[0](dst_data, encoded_mem, self_causal_mask, mem_attn_mask)
    states = dec.layers[0].init_states(batch_size, h_out.ctx, h_out.dtype)
    h_out_from_incremental = []
    for i in range(tgt_seq_length):
        ele_h_out, states = dec.layers[0].incremental_decode(mx, dst_data[:, i:(i + 1), :], states,
                                                             encoded_mem, src_valid_length,
                                                             enc_mem_attn_mask)
        h_out_from_incremental.append(ele_h_out)
    h_out_from_incremental = mx.np.concatenate(h_out_from_incremental, axis=1)

    for i in range(batch_size):
        val_length = dst_valid_length[i].asnumpy()
        assert_allclose(h_out_from_incremental[i, :val_length, :].asnumpy(),
                        h_out[i, :val_length, :].asnumpy(), 1E-5, 1E-5)
    # Test for the full decoder
    states = dec.init_states(batch_size, src_data.ctx, src_data.dtype)
    final_out_from_incremental = []
    for i in range(tgt_seq_length):
        ele_final_out, states = dec.incremental_decode(mx, dst_data[:, i:(i + 1), :],
                                                       states, encoded_mem, src_valid_length)
        final_out_from_incremental.append(ele_final_out)
    final_out_from_incremental = mx.np.concatenate(final_out_from_incremental, axis=1)
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
@pytest.mark.parametrize('tie_weights', [False, True])
def test_transformer_nmt_model(train_hybridize, inference_hybridize,
                               enc_pre_norm, dec_pre_norm,
                               enc_units, dec_units,
                               enc_num_layers, dec_num_layers,
                               enc_recurrent, dec_recurrent, tie_weights):
    src_seq_length = 20
    tgt_seq_length = 15
    src_vocab_size = 32
    tgt_vocab_size = 32
    if enc_units != dec_units:
        shared_embed = False
    else:
        shared_embed = True
    model = TransformerNMTModel(src_vocab_size=src_vocab_size,
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
                                dropout=0.0)
    inference_model = TransformerNMTInference(model=model)
    model.initialize()
    if train_hybridize:
        model.hybridize()
    verify_nmt_model(model)
    if inference_hybridize:
        inference_model.hybridize()
    verify_nmt_inference(train_model=model, inference_model=inference_model)


def test_transformer_cfg_registry():
    assert len(transformer_nmt_cfg_reg.list_keys()) > 0


@pytest.mark.parametrize('cfg_key', transformer_nmt_cfg_reg.list_keys())
def test_transformer_cfg(cfg_key):
    cfg = TransformerNMTModel.get_cfg(cfg_key)
    cfg.defrost()
    cfg.MODEL.src_vocab_size = 1000
    cfg.MODEL.tgt_vocab_size = 1000
    cfg.freeze()
    model = TransformerNMTModel.from_cfg(cfg)
    model.initialize()
    model.hybridize()
    mx.npx.waitall()
