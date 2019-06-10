import os
import io
import json
import argparse
import mxnet as mx
import re
from gluonnlp.vocab import Vocab
from model import GPT2_117M, GPT2_345M


def read_tf_checkpoint(path):
    """read tensorflow checkpoint"""
    from tensorflow.python import pywrap_tensorflow
    tensors = {}
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tensor = reader.get_tensor(key)
        tensors[key] = tensor
    return tensors


def convert_vocab_bpe(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with io.open(os.path.join(src_dir, 'encoder.json'), 'r', encoding='utf-8') as f:
        token_to_idx = json.load(f)
        token_to_idx = {k : int(v) for k, v in token_to_idx.items()}
    idx_to_token = {int(v): k for k, v in token_to_idx.items()}
    idx_to_token = [idx_to_token[i] for i in range(len(idx_to_token))]
    vocab = Vocab(unknown_token=None)
    vocab._idx_to_token = idx_to_token
    vocab._token_to_idx = token_to_idx
    vocab._reserved_tokens = None
    vocab._padding_token = None
    vocab._bos_token = None
    vocab._eos_token = '<|endoftext|>'
    with io.open(os.path.join(dst_dir, 'vocab.json'), 'w', encoding='utf-8') as of:
        of.write(vocab.to_json())
    with io.open(os.path.join(src_dir, 'vocab.bpe'), 'r', encoding='utf-8') as f:
        of = io.open(os.path.join(dst_dir, 'bpe_ranks.json'), 'w', encoding='utf-8')
        of.write(f.read())

def convert_tf_param(gluon_model, tf_ckpt_path, gluon_param_save_path):
    TF_GLUON_NAME_MAP = {
        'model/wte' : '_embed.weight',
        'model/wpe' : '_pos_embed.weight',
        'model/h(\d+)/ln_1/b' : '_attn_ln.{}.beta',
        'model/h(\d+)/ln_1/g' : '_attn_ln.{}.gamma',
        'model/h(\d+)/ln_2/b' : '_ffn_ln.{}.beta',
        'model/h(\d+)/ln_2/g' : '_ffn_ln.{}.gamma',
        'model/h(\d+)/mlp/c_fc/w' : '_ffn_layers.{}._hidden_map.weight',
        'model/h(\d+)/mlp/c_fc/b' : '_ffn_layers.{}._hidden_map.bias',
        'model/h(\d+)/mlp/c_proj/w' : '_ffn_layers.{}._out_map.weight',
        'model/h(\d+)/mlp/c_proj/b' : '_ffn_layers.{}._out_map.bias',
        'model/h(\d+)/attn/c_attn/w' : '_self_attention_layers.{}._multi_head_qkv_proj.weight',
        'model/h(\d+)/attn/c_attn/b' : '_self_attention_layers.{}._multi_head_qkv_proj.bias',
        'model/h(\d+)/attn/c_proj/w' : '_self_attention_layers.{}._out_proj.weight',
        'model/h(\d+)/attn/c_proj/b' : '_self_attention_layers.{}._out_proj.bias',
        'model/ln_f/b' : '_final_ln.beta',
        'model/ln_f/g' : '_final_ln.gamma'
    }
    tf_params = read_tf_checkpoint(tf_ckpt_path)
    gluon_model.initialize()
    out = gluon_model(mx.nd.zeros((1, 1)))
    params = gluon_model._collect_params_with_prefix()
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
            params[gluon_param_name].set_data(mx.nd.array(param_value[0].T))
        else:
            params[gluon_param_name].set_data(param_value)
        loaded[gluon_param_name] = True
    for name in params:
        if not loaded[name] and name != '_logits_proj.weight':
            print('{} is not loaded!'.format(name))
    gluon_model.save_parameters(gluon_param_save_path)
    mx.nd.waitall()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="Source path of the model directory in openai/gpt-2", type=str, required=True)
    parser.add_argument("--dst_dir", help="Destination path of the model directory of gluonnlp", type=str, required=True)
    parser.add_argument('--model', help='The specific model we need to convert', type=str, choices=['117M', '345M'])

    args = parser.parse_args()
    print('Convert {} to {}'.format(os.path.join(args.src_dir, args.model),
                                    os.path.join(args.dst_dir, args.model)))
    convert_vocab_bpe(os.path.join(args.src_dir, args.model),
                      os.path.join(args.dst_dir, args.model))
    if args.model == '117M':
        gluon_model = GPT2_117M()
    elif args.model == '345M':
        gluon_model = GPT2_345M()
    else:
        raise NotImplementedError
    convert_tf_param(gluon_model, os.path.join(args.src_dir, args.model, 'model.ckpt'),
                     os.path.join(args.dst_dir, args.model, 'model.params'))
