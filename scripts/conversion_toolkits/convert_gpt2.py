import os
import re
import json
import shutil
import logging
import argparse

import tensorflow as tf
from tensorflow.contrib.training import HParams
from gpt_2.src import model

import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose

from gluonnlp.data.vocab import Vocab
from gluonnlp.utils.misc import sha1sum, logging_config, naming_convention
from gluonnlp.models.gpt2 import GPT2Model, GPT2ForLM

mx.npx.set_np()


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the tf GPT-2 Model to Gluon.')
    parser.add_argument('--tf_model_path', type=str, required=True,
                        help='Directory of the tf GPT-2 model.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory path to save the converted GPT-2 model.')
    parser.add_argument('--test', action='store_true',
                        help='Whether to test the conversion.')
    return parser.parse_args()

def convert_vocab(args):
    print('converting vocab')
    merges_path = os.path.join(args.tf_model_path, 'vocab.bpe')
    vocab_path = os.path.join(args.tf_model_path, 'encoder.json')
    gluon_merges_path = os.path.join(args.save_dir, 'gpt2.merges')
    gluon_vocab_path = os.path.join(args.save_dir, 'gpt2.vocab')
    
    shutil.copy(merges_path, gluon_merges_path)
    with open(vocab_path, 'r', encoding='utf-8') as f_v:
        tf_vocab = json.load(f_v)
    tf_vocab = list(tf_vocab.items())
    tf_vocab = sorted(tf_vocab, key=lambda x: x[1])
    all_tokens = [e[0] for e in tf_vocab]
    eos_token = all_tokens[-1]
    assert eos_token == '<|endoftext|>'
    gluon_vocab = Vocab(all_tokens,
                        unk_token=None,
                        eos_token=eos_token)
    gluon_vocab.save(gluon_vocab_path)

    vocab_size = len(gluon_vocab)
    print('| converted dictionary: {} types'.format(vocab_size))
    return vocab_size


def convert_config(tf_cfg, vocab_size):
    print('converting config')
    cfg = GPT2Model.get_cfg().clone()
    cfg.defrost()
    cfg.MODEL.vocab_size = tf_cfg['n_vocab']
    cfg.MODEL.units = tf_cfg['n_embd']
    cfg.MODEL.max_length = tf_cfg['n_ctx']
    cfg.MODEL.num_heads = tf_cfg['n_head']
    cfg.MODEL.num_layers = tf_cfg['n_layer']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


def read_tf_ckpt(path):
    from tensorflow.python import pywrap_tensorflow
    tensors = {}
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tensor = reader.get_tensor(key)
        tensors[key] = tensor
    return tensors


def convert_backbone_params(tf_params, gluon_backbone_model):
    TF_GLUON_NAME_MAP = {
        'model/wte' : '_embed.weight',
        'model/wpe' : '_pos_embed._embed.weight',
        'model/h(\d+)/ln_1/b' : '_layers.{}.atten.ln.beta',
        'model/h(\d+)/ln_1/g' : '_layers.{}.atten.ln.gamma',
        'model/h(\d+)/ln_2/b' : '_layers.{}.ffn.layer_norm.beta',
        'model/h(\d+)/ln_2/g' : '_layers.{}.ffn.layer_norm.gamma',
        'model/h(\d+)/mlp/c_fc/w' : '_layers.{}.ffn.ffn_1.weight',
        'model/h(\d+)/mlp/c_fc/b' : '_layers.{}.ffn.ffn_1.bias',
        'model/h(\d+)/mlp/c_proj/w' : '_layers.{}.ffn.ffn_2.weight',
        'model/h(\d+)/mlp/c_proj/b' : '_layers.{}.ffn.ffn_2.bias',
        'model/h(\d+)/attn/c_attn/w' : '_layers.{}.atten.qkv.weight',
        'model/h(\d+)/attn/c_attn/b' : '_layers.{}.atten.qkv.bias',
        'model/h(\d+)/attn/c_proj/w' : '_layers.{}.atten.out_proj.weight',
        'model/h(\d+)/attn/c_proj/b' : '_layers.{}.atten.out_proj.bias',
        'model/ln_f/b' : '_final_ln.beta',
        'model/ln_f/g' : '_final_ln.gamma'
    }
    
    params = gluon_backbone_model.collect_params()
    loaded = {k: False for k in params}
    for name, param_value in tf_params.items():
        gluon_param_name = None
        for lhs, rhs in TF_GLUON_NAME_MAP.items():
            match = re.match(lhs, name)
            if match is not None:
                if len(match.groups()) > 0:
                    gluon_param_name = rhs.format(match.groups()[0])
                    break
                else:
                    gluon_param_name = rhs
        assert gluon_param_name is not None
        print('{} --> {}'.format(name, gluon_param_name))
        if param_value.shape != params[gluon_param_name].shape:
            params[gluon_param_name].set_data(param_value[0].T)
        else:
            params[gluon_param_name].set_data(param_value)
        loaded[gluon_param_name] = True
    for name in params:
        if not loaded[name]:
            print('{} is not loaded!'.format(name))


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


def test_model(tf_model_path, gluon_model):
    # test data
    ctx = mx.cpu()

    seed = 123
    batch_size = 3
    seq_length = 32
    vocab_size = gluon_model._backbone_model._vocab_size
    np.random.seed(seed)
    input_ids = np.random.randint(
        0,
        vocab_size,
        (batch_size, seq_length)
    )

    with open(os.path.join(tf_model_path, 'hparams.json'), 'r') as hf:
        tf_cfg = json.load(hf)
    hparams = HParams(
        n_vocab=tf_cfg['n_vocab'],
        n_ctx=tf_cfg['n_ctx'],
        n_embd=tf_cfg['n_embd'],
        n_head=tf_cfg['n_head'],
        n_layer=tf_cfg['n_layer'],
    )
    tf_start_states = np.zeros((batch_size, hparams.n_layer, 2, hparams.n_head, 0, hparams.n_embd // hparams.n_head))
    gl_start_states = gluon_model.init_states(batch_size, ctx)

    # gluon model
    gl_input_ids = mx.np.array(input_ids, dtype=np.int32, ctx=ctx)
    gl_logits_1, gl_states = gluon_model(gl_input_ids, gl_start_states)
    gl_logits_2, _ = gluon_model(gl_input_ids, gl_states)

    # tf model
    with tf.Session(graph=tf.Graph()) as sess:    
        tf.set_random_seed(None)
        tf_context = tf.placeholder(tf.int32, [batch_size, seq_length])
        tf_past = tf.placeholder(tf.float32, [batch_size, hparams.n_layer, 2, hparams.n_head,
                                            None, hparams.n_embd // hparams.n_head])
        tf_lm_output = model.model(hparams=hparams, X=tf_context, past=tf_past, reuse=tf.AUTO_REUSE)
        
        tf_saver = tf.train.Saver()
        tf_ckpt = tf.train.latest_checkpoint(tf_model_path)
        tf_saver.restore(sess, tf_ckpt)
        
        tf_output_1 = sess.run(tf_lm_output, feed_dict={tf_context:input_ids, tf_past:tf_start_states})
        tf_logits_1 = tf_output_1['logits']
        tf_present = tf_output_1['present']
        
        tf_output_2 = sess.run(tf_lm_output, feed_dict={tf_context:input_ids, tf_past:tf_present})
        tf_logits_2 = tf_output_2['logits']

    for j in range(batch_size):
        assert_allclose(
            gl_logits_1[j, :, :].asnumpy(),
            tf_logits_1[j, :, :],
            1E-3,
            1E-3
        )
    for j in range(batch_size):
        assert_allclose(
            gl_logits_2[j, :, :].asnumpy(),
            tf_logits_2[j, :, :],
            1E-3,
            1E-3
        )

def convert_gpt2(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    tf_ckpt_path = os.path.join(args.tf_model_path, 'model.ckpt')
    tf_params = read_tf_ckpt(tf_ckpt_path)
    with open(os.path.join(args.tf_model_path, 'hparams.json'), 'r') as hf:
        tf_cfg = json.load(hf)
    
    vocab_size = convert_vocab(args)
    gluon_backbone_cfg = convert_config(tf_cfg, vocab_size)
    with open(os.path.join(args.save_dir, 'model.yml'), 'w') as of:
        of.write(gluon_backbone_cfg.dump())

    gluon_gpt2forlm_model = GPT2ForLM(gluon_backbone_cfg)
    gluon_gpt2forlm_model.initialize(ctx=mx.cpu())
    gluon_gpt2forlm_model.hybridize()
    gluon_backbone_model = gluon_gpt2forlm_model._backbone_model
    convert_backbone_params(tf_params, gluon_backbone_model)
    
    if args.test:
        test_model(args.tf_model_path, gluon_gpt2forlm_model)

    gluon_gpt2forlm_model.save_parameters(os.path.join(args.save_dir, 'model_lm.params'), deduplicate=True)
    logging.info('Convert the GPT2 LM model in {} to {}'.
                 format(os.path.join(args.tf_model_path, 'model.ckpt'),
                        os.path.join(args.save_dir, 'model_lm.params')))
    gluon_backbone_model.save_parameters(os.path.join(args.save_dir, 'model.params'), deduplicate=True)
    logging.info('Convert the GPT2 backbone model in {} to {}'.
                 format(os.path.join(args.tf_model_path, 'model.ckpt'),
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
    convert_gpt2(args)
