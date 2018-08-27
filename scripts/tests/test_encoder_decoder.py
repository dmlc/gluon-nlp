import numpy as np
import mxnet as mx
from mxnet.test_utils import assert_almost_equal
from ..nmt.gnmt import *
from ..nmt.transformer import *


def test_gnmt_encoder():
    ctx = mx.Context.default_ctx
    for cell_type in ["lstm", "gru", "relu_rnn", "tanh_rnn"]:
        for num_layers, num_bi_layers in [(2, 1), (3, 0)]:
            for use_residual in [False, True]:
                encoder = GNMTEncoder(cell_type=cell_type, num_layers=num_layers,
                                      num_bi_layers=num_bi_layers, hidden_size=8,
                                      dropout=0.0, use_residual=use_residual,
                                      prefix='gnmt_encoder_')
                encoder.initialize(ctx=ctx)
                encoder.hybridize()
                for batch_size in [4]:
                    for seq_length in [5, 10]:
                        inputs_nd = mx.nd.random.normal(0, 1, shape=(batch_size, seq_length, 4), ctx=ctx)
                        valid_length_nd = mx.nd.array(np.random.randint(1, seq_length,
                                                                        size=(batch_size,)), ctx=ctx)
                        encoder_outputs, _ = encoder(inputs_nd, valid_length=valid_length_nd)
                        valid_length_npy = valid_length_nd.asnumpy()
                        rnn_output = encoder_outputs[0].asnumpy()
                        for i in range(batch_size):
                            if valid_length_npy[i] < seq_length - 1:
                                padded_out = rnn_output[i, int(valid_length_npy[i]):, :]
                                assert_almost_equal(padded_out, np.zeros_like(padded_out), 1E-6, 1E-6)
                        assert(encoder_outputs[0].shape == (batch_size, seq_length, 8))
                        assert(len(encoder_outputs[1]) == num_layers)


def test_gnmt_encoder_decoder():
    ctx = mx.Context.default_ctx
    num_hidden = 8
    encoder = GNMTEncoder(cell_type="lstm", num_layers=3, num_bi_layers=1, hidden_size=num_hidden,
                          dropout=0.0, use_residual=True, prefix='gnmt_encoder_')
    encoder.initialize(ctx=ctx)
    encoder.hybridize()
    for output_attention in [True, False]:
        for use_residual in [True, False]:
            decoder = GNMTDecoder(cell_type="lstm", num_layers=3, hidden_size=num_hidden, dropout=0.0,
                                  output_attention=output_attention, use_residual=use_residual, prefix='gnmt_decoder_')
            decoder.initialize(ctx=ctx)
            decoder.hybridize()
            for batch_size in [4]:
                for src_seq_length, tgt_seq_length in [(5, 10), (10, 5)]:
                    src_seq_nd = mx.nd.random.normal(0, 1, shape=(batch_size, src_seq_length, 4), ctx=ctx)
                    tgt_seq_nd = mx.nd.random.normal(0, 1, shape=(batch_size, tgt_seq_length, 4), ctx=ctx)
                    src_valid_length_nd = mx.nd.array(np.random.randint(1, src_seq_length, size=(batch_size,)), ctx=ctx)
                    tgt_valid_length_nd = mx.nd.array(np.random.randint(1, tgt_seq_length, size=(batch_size,)), ctx=ctx)
                    src_valid_length_npy = src_valid_length_nd.asnumpy()
                    tgt_valid_length_npy = tgt_valid_length_nd.asnumpy()
                    encoder_outputs, _ = encoder(src_seq_nd, valid_length=src_valid_length_nd)
                    decoder_states = decoder.init_state_from_encoder(encoder_outputs, src_valid_length_nd)

                    # Test multi step forwarding
                    output, new_states, additional_outputs = decoder.decode_seq(tgt_seq_nd,
                                                                                decoder_states,
                                                                                tgt_valid_length_nd)
                    assert(output.shape == (batch_size, tgt_seq_length, num_hidden))
                    output_npy = output.asnumpy()
                    for i in range(batch_size):
                        tgt_v_len = int(tgt_valid_length_npy[i])
                        if tgt_v_len < tgt_seq_length - 1:
                            assert((output_npy[i, tgt_v_len:, :] == 0).all())
                    if output_attention:
                        assert(len(additional_outputs) == 1)
                        attention_out = additional_outputs[0].asnumpy()
                        assert(attention_out.shape == (batch_size, tgt_seq_length, src_seq_length))
                        for i in range(batch_size):
                            mem_v_len = int(src_valid_length_npy[i])
                            if mem_v_len < src_seq_length - 1:
                                assert((attention_out[i, :, mem_v_len:] == 0).all())
                            if mem_v_len > 0:
                                assert_almost_equal(attention_out[i, :, :].sum(axis=-1),
                                                    np.ones(attention_out.shape[1]))
                    else:
                        assert(len(additional_outputs) == 0)

def test_transformer_encoder():
    ctx = mx.Context.default_ctx
    for num_layers in range(1, 3):
        for output_attention in [True, False]:
            for use_residual in [False, True]:
                encoder = TransformerEncoder(num_layers=num_layers, max_length=10,
                                             units=16, hidden_size=32, num_heads=8,
                                             dropout=0.0, use_residual=use_residual,
                                             output_attention=output_attention, prefix='transformer_encoder_')
                encoder.initialize(ctx=ctx)
                encoder.hybridize()
                for batch_size in [4]:
                    for seq_length in [5, 10]:
                        inputs_nd = mx.nd.random.normal(0, 1, shape=(batch_size, seq_length, 16), ctx=ctx)
                        valid_length_nd = mx.nd.array(np.random.randint(1, seq_length,
                                                                        size=(batch_size,)), ctx=ctx)
                        encoder_outputs, additional_outputs = encoder(inputs_nd, valid_length=valid_length_nd)
                        valid_length_npy = valid_length_nd.asnumpy()
                        encoder_outputs = encoder_outputs.asnumpy()
                        for i in range(batch_size):
                            if valid_length_npy[i] < seq_length - 1:
                                padded_out = encoder_outputs[i, int(valid_length_npy[i]):, :]
                                assert_almost_equal(padded_out, np.zeros_like(padded_out), 1E-6, 1E-6)
                        assert(encoder_outputs.shape == (batch_size, seq_length, 16))
                        if output_attention:
                            assert(len(additional_outputs) == num_layers)
                            attention_out = additional_outputs[0][0].asnumpy()
                            assert(attention_out.shape == (batch_size, 8, seq_length, seq_length))
                            for i in range(batch_size):
                                mem_v_len = int(valid_length_npy[i])
                                if mem_v_len < seq_length - 1:
                                    assert((attention_out[i, :, :, mem_v_len:] == 0).all())
                                if mem_v_len > 0:
                                    assert_almost_equal(attention_out[i, :, :, :].sum(axis=-1),
                                                      np.ones(attention_out.shape[1:3]))
                        else:
                            assert(len(additional_outputs) == 0)

def test_transformer_encoder_decoder():
    ctx = mx.Context.default_ctx
    units = 16
    encoder = TransformerEncoder(num_layers=3, units=units, hidden_size=32, num_heads=8, max_length=10,
                                 dropout=0.0, use_residual=True, prefix='transformer_encoder_')
    encoder.initialize(ctx=ctx)
    encoder.hybridize()
    for output_attention in [True, False]:
        for use_residual in [True, False]:
            decoder = TransformerDecoder(num_layers=3, units=units, hidden_size=32, num_heads=8, max_length=10, dropout=0.0,
                                         output_attention=output_attention, use_residual=use_residual, prefix='transformer_decoder_')
            decoder.initialize(ctx=ctx)
            decoder.hybridize()
            for batch_size in [4]:
                for src_seq_length, tgt_seq_length in [(5, 10), (10, 5)]:
                    src_seq_nd = mx.nd.random.normal(0, 1, shape=(batch_size, src_seq_length, units), ctx=ctx)
                    tgt_seq_nd = mx.nd.random.normal(0, 1, shape=(batch_size, tgt_seq_length, units), ctx=ctx)
                    src_valid_length_nd = mx.nd.array(np.random.randint(1, src_seq_length, size=(batch_size,)), ctx=ctx)
                    tgt_valid_length_nd = mx.nd.array(np.random.randint(1, tgt_seq_length, size=(batch_size,)), ctx=ctx)
                    src_valid_length_npy = src_valid_length_nd.asnumpy()
                    tgt_valid_length_npy = tgt_valid_length_nd.asnumpy()
                    encoder_outputs, _ = encoder(src_seq_nd, valid_length=src_valid_length_nd)
                    decoder_states = decoder.init_state_from_encoder(encoder_outputs, src_valid_length_nd)

                    # Test multi step forwarding
                    output, new_states, additional_outputs = decoder.decode_seq(tgt_seq_nd,
                                                                                decoder_states,
                                                                                tgt_valid_length_nd)
                    assert(output.shape == (batch_size, tgt_seq_length, units))
                    output_npy = output.asnumpy()
                    for i in range(batch_size):
                        tgt_v_len = int(tgt_valid_length_npy[i])
                        if tgt_v_len < tgt_seq_length - 1:
                            assert((output_npy[i, tgt_v_len:, :] == 0).all())
                    if output_attention:
                        assert(len(additional_outputs) == 3)
                        attention_out = additional_outputs[0][1].asnumpy()
                        assert(attention_out.shape == (batch_size, 8, tgt_seq_length, src_seq_length))
                        for i in range(batch_size):
                            mem_v_len = int(src_valid_length_npy[i])
                            if mem_v_len < src_seq_length - 1:
                                assert((attention_out[i, :, :, mem_v_len:] == 0).all())
                            if mem_v_len > 0:
                                assert_almost_equal(attention_out[i, :, :, :].sum(axis=-1),
                                                    np.ones(attention_out.shape[1:3]))
                    else:
                        assert(len(additional_outputs) == 0)

