import os
import shutil
import logging
import argparse

import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose

import torch
from fairseq.models.bart import BARTModel as fairseq_BARTModel
from gluonnlp.utils.misc import sha1sum, logging_config, naming_convention
from gluonnlp.models.bart import BartModel
from convert_fairseq_roberta import convert_vocab

mx.npx.set_np()


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the fairseq BART Model to Gluon.')
    parser.add_argument('--fairseq_model_path', type=str, required=True,
                        help='Directory of the fairseq BART model.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory path to save the converted BART model.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='The single gpu to run mxnet, (e.g. --gpu 0) the default device is cpu.')
    parser.add_argument('--test', action='store_true',
                        help='Whether to test the conversion.')
    return parser.parse_args()


def convert_config(fairseq_cfg, vocab_size, cfg):
    print('converting config')
    cfg.defrost()
    # Config for the bart base model
    cfg.MODEL.vocab_size = vocab_size
    cfg.MODEL.max_src_length = fairseq_cfg.max_source_positions
    cfg.MODEL.max_tgt_length = fairseq_cfg.max_target_positions
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.shared_embed = fairseq_cfg.share_all_embeddings
    cfg.MODEL.scale_embed = not fairseq_cfg.no_scale_embedding
    cfg.MODEL.tie_weights = fairseq_cfg.share_decoder_input_output_embed
    cfg.MODEL.data_norm = fairseq_cfg.layernorm_embedding
    cfg.MODEL.pooler_activation = fairseq_cfg.pooler_activation_fn
    cfg.MODEL.layer_norm_eps = 1E-5
    cfg.MODEL.dropout = fairseq_cfg.dropout
    cfg.MODEL.activation_dropout = fairseq_cfg.activation_dropout
    cfg.MODEL.attention_dropout = fairseq_cfg.attention_dropout
    cfg.MODEL.dtype = 'float32'

    # Parameters for the encoder
    cfg.MODEL.ENCODER.pre_norm = fairseq_cfg.encoder_normalize_before
    cfg.MODEL.ENCODER.num_layers = fairseq_cfg.encoder_layers
    cfg.MODEL.ENCODER.units = fairseq_cfg.encoder_embed_dim
    cfg.MODEL.ENCODER.num_heads = fairseq_cfg.encoder_attention_heads
    cfg.MODEL.ENCODER.hidden_size = fairseq_cfg.encoder_ffn_embed_dim
    cfg.MODEL.ENCODER.activation = fairseq_cfg.activation_fn

    # Parameters for the decoder
    cfg.MODEL.DECODER.pre_norm = fairseq_cfg.decoder_normalize_before
    cfg.MODEL.DECODER.num_layers = fairseq_cfg.decoder_layers
    cfg.MODEL.DECODER.units = fairseq_cfg.decoder_embed_dim
    cfg.MODEL.DECODER.num_heads = fairseq_cfg.decoder_attention_heads
    cfg.MODEL.DECODER.hidden_size = fairseq_cfg.decoder_ffn_embed_dim
    cfg.MODEL.DECODER.activation = fairseq_cfg.activation_fn

    cfg.INITIALIZER.embed = ['xavier', 'gaussian', 'in', 1.0]
    cfg.INITIALIZER.weight = ['xavier', 'uniform', 'avg', 1.0]
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


def convert_params(fairseq_model,
                   gluon_cfg,
                   ctx):
    fairseq_params = fairseq_model.state_dict()
    # apply a linear mapping to vocab dictionary
    gluon_model = BartModel.from_cfg(gluon_cfg, use_pooler=False)
    gluon_model.initialize(ctx=ctx)
    gluon_model.hybridize()
    gluon_params = gluon_model.collect_params()
    all_keys = set(gluon_params.keys())

    def convert_attention(num_layers,
                          fairseq_prefix,
                          gluon_prefix,
                          fairseq_attn_prefix='self_attn',
                          gluon_attn_prefix='attn_qkv'):
        for layer_id in range(num_layers):
            fs_atten_prefix = \
                '{}.layers.{}.{}.' \
                .format(fairseq_prefix, layer_id, fairseq_attn_prefix)
            fs_q_weight = fairseq_params[fs_atten_prefix + 'q_proj.weight'].cpu().numpy()
            fs_k_weight = fairseq_params[fs_atten_prefix + 'k_proj.weight'].cpu().numpy()
            fs_v_weight = fairseq_params[fs_atten_prefix + 'v_proj.weight'].cpu().numpy()
            fs_q_bias = fairseq_params[fs_atten_prefix + 'q_proj.bias'].cpu().numpy()
            fs_k_bias = fairseq_params[fs_atten_prefix + 'k_proj.bias'].cpu().numpy()
            fs_v_bias = fairseq_params[fs_atten_prefix + 'v_proj.bias'].cpu().numpy()
            gl_qkv_prefix = \
                '{}.layers.{}.{}.' \
                .format(gluon_prefix, layer_id, gluon_attn_prefix)
            gl_qkv_weight = gluon_params[gl_qkv_prefix + 'weight']
            gl_qkv_bias = gluon_params[gl_qkv_prefix + 'bias']
            all_keys.remove(gl_qkv_prefix + 'weight')
            all_keys.remove(gl_qkv_prefix + 'bias')
            gl_qkv_weight.set_data(
                np.concatenate([fs_q_weight, fs_k_weight, fs_v_weight], axis=0))
            gl_qkv_bias.set_data(
                np.concatenate([fs_q_bias, fs_k_bias, fs_v_bias], axis=0))

    def convert_ffn(num_layers, fairseq_prefix, gluon_prefix):
        # convert feed forward layer in encoder
        for layer_id in range(num_layers):
            for k, v in [
                ('fc1.weight', 'ffn.ffn_1.weight'),
                ('fc1.bias', 'ffn.ffn_1.bias'),
                ('fc2.weight', 'ffn.ffn_2.weight'),
                ('fc2.bias', 'ffn.ffn_2.bias'),
                ('final_layer_norm.weight', 'ffn.layer_norm.gamma'),
                ('final_layer_norm.bias', 'ffn.layer_norm.beta')
            ]:
                fs_name = '{}.layers.{}.{}' \
                          .format(fairseq_prefix, layer_id, k)
                gl_name = '{}.layers.{}.{}' \
                          .format(gluon_prefix, layer_id, v)
                all_keys.remove(gl_name)
                gluon_params[gl_name].set_data(
                    fairseq_params[fs_name].cpu().numpy())

    print('converting embedding params')
    padding_idx = fairseq_model.task.dictionary.pad_index
    for fs_name, gl_name in [
        ('model.encoder.embed_tokens.weight', 'src_embed_layer.weight'),
        ('model.encoder.embed_positions.weight', 'src_pos_embed_layer._embed.weight'),
        ('model.encoder.layernorm_embedding.weight', 'encoder.ln_data.gamma'),
        ('model.encoder.layernorm_embedding.bias', 'encoder.ln_data.beta'),
        ('model.decoder.embed_tokens.weight', 'tgt_embed_layer.weight'),
        ('model.decoder.embed_positions.weight', 'tgt_pos_embed_layer._embed.weight'),
        ('model.decoder.layernorm_embedding.weight', 'decoder.ln_data.gamma'),
        ('model.decoder.layernorm_embedding.bias', 'decoder.ln_data.beta'),
        # final projection in decoder
        ('model.decoder.output_projection.weight', 'tgt_final_layer.weight'),
    ]:
        all_keys.remove(gl_name)
        if 'embed_positions' in fs_name:
            # position embed weight
            gluon_params[gl_name].set_data(
                fairseq_params[fs_name].cpu().numpy()[padding_idx + 1:, :])
        else:
            gluon_params[gl_name].set_data(
                fairseq_params[fs_name].cpu().numpy())

    print('converting encoder params')
    encoder_num_layers = gluon_cfg.MODEL.ENCODER.num_layers
    convert_attention(encoder_num_layers, 'model.encoder', 'encoder')
    convert_ffn(encoder_num_layers, 'model.encoder', 'encoder')
    for layer_id in range(encoder_num_layers):
        for k, v in [
            ('self_attn.out_proj.weight', 'attention_proj.weight'),
            ('self_attn.out_proj.bias', 'attention_proj.bias'),
            ('self_attn_layer_norm.weight', 'layer_norm.gamma'),
            ('self_attn_layer_norm.bias', 'layer_norm.beta'),
        ]:
            fs_name = 'model.encoder.layers.{}.{}' \
                      .format(layer_id, k)
            gl_name = 'encoder.layers.{}.{}' \
                      .format(layer_id, v)
            all_keys.remove(gl_name)
            gluon_params[gl_name].set_data(
                fairseq_params[fs_name].cpu().numpy())

    print('converting decoder params')
    decoder_num_layers = gluon_cfg.MODEL.DECODER.num_layers
    convert_attention(decoder_num_layers, 'model.decoder', 'decoder',
                      gluon_attn_prefix='attn_in_qkv')
    convert_ffn(decoder_num_layers, 'model.decoder', 'decoder')

    for layer_id in range(decoder_num_layers):
        for k, v in [
            ('self_attn.out_proj.weight', 'proj_in.weight'),
            ('self_attn.out_proj.bias', 'proj_in.bias'),
            ('self_attn_layer_norm.weight', 'ln_in.gamma'),
            ('self_attn_layer_norm.bias', 'ln_in.beta'),
            ('encoder_attn.out_proj.weight', 'proj_inter.weight'),
            ('encoder_attn.out_proj.bias', 'proj_inter.bias'),
            ('encoder_attn_layer_norm.weight', 'ln_inter.gamma'),
            ('encoder_attn_layer_norm.bias', 'ln_inter.beta'),
            ('encoder_attn.q_proj.weight', 'attn_inter_q.weight'),
            ('encoder_attn.q_proj.bias', 'attn_inter_q.bias'),
            ('encoder_attn.k_proj.weight', 'attn_inter_k.weight'),
            ('encoder_attn.k_proj.bias', 'attn_inter_k.bias'),
            ('encoder_attn.v_proj.weight', 'attn_inter_v.weight'),
            ('encoder_attn.v_proj.bias', 'attn_inter_v.bias'),

        ]:
            fs_name = 'model.decoder.layers.{}.{}' \
                      .format(layer_id, k)
            gl_name = 'decoder.layers.{}.{}' \
                      .format(layer_id, v)
            all_keys.remove(gl_name)
            gluon_params[gl_name].set_data(
                fairseq_params[fs_name].cpu().numpy())

    assert len(all_keys) == 0, 'parameters missing from tensorflow checkpoint'

    # check parameters sharing if share_decoder_input_output_embed is true
    assert np.array_equal(
        fairseq_params['model.decoder.embed_tokens.weight'].cpu().numpy(),
        fairseq_params['model.decoder.output_projection.weight'].cpu().numpy()
    )
    return gluon_model


def test_model(fairseq_model, gluon_model, gpu):
    print('testing model')
    ctx = mx.gpu(gpu) if gpu is not None else mx.cpu()
    batch_size = 3
    seq_length = 32
    vocab_size = len(fairseq_model.task.dictionary)
    padding_id = fairseq_model.model.decoder.padding_idx
    input_ids = np.random.randint(  # skip padding_id
        padding_id + 1,
        vocab_size,
        (batch_size, seq_length)
    )
    valid_length = np.random.randint(
        seq_length // 2,
        seq_length,
        (batch_size,)
    )

    for i in range(batch_size):  # add padding, for fairseq padding mask
        input_ids[i, valid_length[i]:] = padding_id

    gl_input_ids = mx.np.array(input_ids, dtype=np.int32, ctx=ctx)
    gl_valid_length = mx.np.array(valid_length, dtype=np.int32, ctx=ctx)
    gl_dec_out = \
        gluon_model(gl_input_ids, gl_valid_length, gl_input_ids, gl_valid_length)

    fs_input_ids = torch.from_numpy(input_ids).cuda(gpu)
    fairseq_model.model.eval()
    fs_dec_out, fs_extra = \
        fairseq_model.model.cuda(gpu)(
            fs_input_ids,
            valid_length,
            fs_input_ids,
            return_all_hiddens=True)

    # checking decoder output
    gl_dec_out = gl_dec_out.asnumpy()
    fs_dec_out = fs_dec_out.detach().cpu().numpy()
    for j in range(batch_size):
        assert_allclose(
            gl_dec_out[j, :valid_length[j], :],
            fs_dec_out[j, :valid_length[j], :],
            1E-3,
            1E-3
        )


def rename(save_dir):
    """Rename converted files with hash"""
    old_names = os.listdir(save_dir)
    for old_name in old_names:
        old_path = os.path.join(save_dir, old_name)
        long_hash = sha1sum(old_path)
        file_prefix, file_sufix = old_name.split('.')
        new_name = '{file_prefix}-{short_hash}.{file_sufix}'.format(
            file_prefix=file_prefix,
            short_hash=long_hash[:8],
            file_sufix=file_sufix)
        new_path = os.path.join(save_dir, new_name)
        shutil.move(old_path, new_path)
        file_size = os.path.getsize(new_path)
        logging.info('\t{} {} {}'.format(new_path, long_hash, file_size))


def convert_fairseq_model(args):
    if not args.save_dir:
        args.save_dir = os.path.basename(args.fairseq_model_path) + '_gluon'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    fairseq_bart = fairseq_BARTModel.from_pretrained(args.fairseq_model_path,
                                                     checkpoint_file='model.pt')
    vocab_size = convert_vocab(args, fairseq_bart)
    gluon_cfg = convert_config(fairseq_bart.cfg.model, vocab_size,
                               BartModel.get_cfg().clone())
    with open(os.path.join(args.save_dir, 'model.yml'), 'w') as of:
        of.write(gluon_cfg.dump())

    ctx = mx.gpu(args.gpu) if args.gpu is not None else mx.cpu()
    gluon_bart = convert_params(fairseq_bart,
                                gluon_cfg,
                                ctx)
    if args.test:
        test_model(fairseq_bart, gluon_bart, args.gpu)

    gluon_bart.save_parameters(os.path.join(args.save_dir, 'model.params'))
    logging.info('Convert the BART MLM model in {} to {}'.
                 format(os.path.join(args.fairseq_model_path, 'model.pt'),
                        os.path.join(args.save_dir, 'model.params')))

    logging.info('Conversion finished!')
    logging.info('Statistics:')
    old_names = os.listdir(args.save_dir)
    for old_name in old_names:
        new_name, long_hash = naming_convention(args.save_dir, old_name)
        old_path = os.path.join(args.save_dir, old_name)
        new_path = os.path.join(args.save_dir, new_name)
        shutil.move(old_path, new_path)
        file_size = os.path.getsize(new_path)
        logging.info('\t{}/{} {} {}'.format(args.save_dir, new_name, long_hash, file_size))


if __name__ == '__main__':
    args = parse_args()
    logging_config()
    convert_fairseq_model(args)
