#!/usr/bin/env python
# encoding: utf-8
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import argparse
import logging
import json
import mxnet as mx
import gluonnlp as nlp
import paddle.fluid as fluid

from gluonnlp.model import BERTEncoder, BERTModel
from gluonnlp.model.bert import bert_hparams
from utils import get_hash, tf_vocab_to_gluon_vocab, load_text_vocab

parser = argparse.ArgumentParser()
parser.add_argument("--gluon_bert_model_base", default='ernie_12_768_12', type=str, help=".")
parser.add_argument("--init_pretraining_params", default='./ERNIE_stable-1.0.1/params',
                    type=str, help=".")
parser.add_argument("--ernie_config_path", default='./ERNIE_stable-1.0.1/ernie_config.json',
                    type=str, help=".")
parser.add_argument("--ernie_vocab_path", default='./ERNIE_stable-1.0.1/vocab.txt',
                    type=str, help=".")
parser.add_argument("--out_dir", default='./ernie_gluon_model2', type=str, help=".")
parser.add_argument("--baidu_lark_repo_dir", default='../../../../LARK', type=str,
                    help='path to the original baidu lark repository. '
                         'The repo should be at f97e3c8581e36dc1979560d62f75df862acd9585.'
                         '(https://github.com/PaddlePaddle/LARK.git)')
args = parser.parse_args()

sys.path = [os.path.join(args.baidu_lark_repo_dir,'ERNIE')] + sys.path
try:
    from model.ernie import ErnieConfig
    from finetune.classifier import create_model
except:
    raise ImportError('Place clone ERNIE first')

def if_exist(var):
    return os.path.exists(os.path.join(args.init_pretraining_params, var.name))


def build_weight_map():
    weight_map = collections.OrderedDict({
        'word_embedding': 'word_embed.0.weight',
        'pos_embedding': 'encoder.position_weight',
        'sent_embedding': 'token_type_embed.0.weight',
        'pre_encoder_layer_norm_scale': 'encoder.layer_norm.gamma',
        'pre_encoder_layer_norm_bias': 'encoder.layer_norm.beta',
    })

    def add_w_and_b(ernie_pre, gluon_pre):
        weight_map[ernie_pre + ".w_0"] = gluon_pre + ".weight"
        weight_map[ernie_pre + ".b_0"] = gluon_pre + ".bias"

    def add_one_encoder_layer(layer_number):
        # attention
        add_w_and_b("encoder_layer_{}_multi_head_att_query_fc".format(layer_number),
                    "encoder.transformer_cells.{}.attention_cell.proj_query".format(layer_number))
        add_w_and_b("encoder_layer_{}_multi_head_att_key_fc".format(layer_number),
                    "encoder.transformer_cells.{}.attention_cell.proj_key".format(layer_number))
        add_w_and_b("encoder_layer_{}_multi_head_att_value_fc".format(layer_number),
                    "encoder.transformer_cells.{}.attention_cell.proj_value".format(layer_number))
        add_w_and_b("encoder_layer_{}_multi_head_att_output_fc".format(layer_number),
                    "encoder.transformer_cells.{}.proj".format(layer_number))
        weight_map["encoder_layer_{}_post_att_layer_norm_bias".format(layer_number)] = \
            "encoder.transformer_cells.{}.layer_norm.beta".format(layer_number)
        weight_map["encoder_layer_{}_post_att_layer_norm_scale".format(layer_number)] = \
            "encoder.transformer_cells.{}.layer_norm.gamma".format(layer_number)
        # intermediate
        add_w_and_b("encoder_layer_{}_ffn_fc_0".format(layer_number),
                    "encoder.transformer_cells.{}.ffn.ffn_1".format(layer_number))
        # output
        add_w_and_b("encoder_layer_{}_ffn_fc_1".format(layer_number),
                    "encoder.transformer_cells.{}.ffn.ffn_2".format(layer_number))
        weight_map["encoder_layer_{}_post_ffn_layer_norm_bias".format(layer_number)] = \
            "encoder.transformer_cells.{}.ffn.layer_norm.beta".format(layer_number)
        weight_map["encoder_layer_{}_post_ffn_layer_norm_scale".format(layer_number)] = \
            "encoder.transformer_cells.{}.ffn.layer_norm.gamma".format(layer_number)

    for i in range(12):
        add_one_encoder_layer(i)
    add_w_and_b('pooled_fc', 'pooler')
    return weight_map


def extract_weights(args):
    # add ERNIE to environment
    print('extract weights start'.center(60, '='))
    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    args.max_seq_len = 512
    args.use_fp16 = False
    args.num_labels = 2
    args.loss_scaling = 1.0
    print('model config:')
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            _, _ = create_model(
                args,
                pyreader_name='train',
                ernie_config=ernie_config)
    fluid.io.load_vars(exe, args.init_pretraining_params, main_program=test_prog, predicate=if_exist)
    state_dict = collections.OrderedDict()
    weight_map = build_weight_map()
    for ernie_name, gluon_name in weight_map.items():
        fluid_tensor = fluid.global_scope().find_var(ernie_name).get_tensor()
        fluid_array = np.array(fluid_tensor, dtype=np.float32)
        if 'w_0' in ernie_name:
            fluid_array = fluid_array.transpose()
        state_dict[gluon_name] = fluid_array
        print(f'{ernie_name} -> {gluon_name} {fluid_array.shape}')
    print('extract weights done!'.center(60, '='))
    return state_dict


def save_model(new_gluon_parameters, output_dir):
    print('save model start'.center(60, '='))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save model
    # load vocab
    vocab_f = open(os.path.join(output_dir, "vocab.txt"), "wt", encoding='utf-8')
    with open(args.ernie_vocab_path, "rt", encoding='utf-8') as f:
        for line in f:
            data = line.strip().split("\t")
            vocab_f.writelines(data[0] + "\n")
    vocab_f.close()
    vocab = tf_vocab_to_gluon_vocab(load_text_vocab(os.path.join(output_dir, "vocab.txt")))
    # vocab serialization
    tmp_file_path = os.path.expanduser(os.path.join(output_dir, 'tmp'))
    if not os.path.exists(os.path.join(args.out_dir)):
        os.makedirs(os.path.join(args.out_dir))
    with open(tmp_file_path, 'w') as f:
        f.write(vocab.to_json())
    hash_full, hash_short = get_hash(tmp_file_path)
    gluon_vocab_path = os.path.expanduser(os.path.join(output_dir, hash_short + '.vocab'))
    with open(gluon_vocab_path, 'w') as f:
        f.write(vocab.to_json())
        logging.info('vocab file saved to %s. hash = %s', gluon_vocab_path, hash_full)

    # BERT config
    tf_config_names_to_gluon_config_names = {
        'attention_probs_dropout_prob': 'embed_dropout',
        'hidden_act': None,
        'hidden_dropout_prob': 'dropout',
        'hidden_size': 'units',
        'initializer_range': None,
        # 'intermediate_size': 'hidden_size',
        'max_position_embeddings': 'max_length',
        'num_attention_heads': 'num_heads',
        'num_hidden_layers': 'num_layers',
        'type_vocab_size': 'token_type_vocab_size',
        'vocab_size': None
    }
    predefined_args = bert_hparams[args.gluon_bert_model_base]
    with open(args.ernie_config_path, 'r') as f:
        tf_config = json.load(f)
        if 'layer_norm_eps' in tf_config: # ignore layer_norm_eps
            del tf_config['layer_norm_eps']
        assert len(tf_config) == len(tf_config_names_to_gluon_config_names)
        for tf_name, gluon_name in tf_config_names_to_gluon_config_names.items():
            if tf_name is None or gluon_name is None:
                continue
            if gluon_name != 'max_length':
                assert tf_config[tf_name] == predefined_args[gluon_name]

    encoder = BERTEncoder(attention_cell=predefined_args['attention_cell'],
                          num_layers=predefined_args['num_layers'], units=predefined_args['units'],
                          hidden_size=predefined_args['hidden_size'],
                          max_length=predefined_args['max_length'],
                          num_heads=predefined_args['num_heads'], scaled=predefined_args['scaled'],
                          dropout=predefined_args['dropout'],
                          use_residual=predefined_args['use_residual'],
                          activation='relu')

    bert = BERTModel(encoder, len(vocab),
                     token_type_vocab_size=predefined_args['token_type_vocab_size'],
                     units=predefined_args['units'], embed_size=predefined_args['embed_size'],
                     embed_dropout=predefined_args['embed_dropout'],
                     word_embed=predefined_args['word_embed'], use_pooler=True,
                     use_decoder=False, use_classifier=False)

    bert.initialize(init=mx.init.Normal(0.02))

    ones = mx.nd.ones((2, 8))
    out = bert(ones, ones, mx.nd.array([5, 6]), mx.nd.array([[1], [2]]))
    params = bert._collect_params_with_prefix()
    assert len(params) == len(new_gluon_parameters), "Gluon model does not match paddle model. " \
                                                   "Please fix the BERTModel hyperparameters"

    # post processings for parameters:
    # - handle tied decoder weight
    new_gluon_parameters['decoder.3.weight'] = new_gluon_parameters['word_embed.0.weight']
    # set parameter data
    loaded_params = {}
    for name in params:
        if name == 'word_embed.0.weight':
            arr = mx.nd.array(new_gluon_parameters[name][:params[name].shape[0]])
        else:
            arr = mx.nd.array(new_gluon_parameters[name])
        try:
            assert arr.shape == params[name].shape
        except:
            print(name)
        params[name].set_data(arr)
        loaded_params[name] = True

    # post processings for parameters:
    # - handle tied decoder weight
    # - update word embedding for reserved tokens

    if len(params) != len(loaded_params):
        raise RuntimeError('The Gluon BERTModel comprises {} parameter arrays, '
                           'but {} have been extracted from the paddle model. '.format(
            len(params), len(loaded_params)))

    # param serialization
    bert.save_parameters(tmp_file_path)
    hash_full, hash_short = get_hash(tmp_file_path)
    gluon_param_path = os.path.expanduser(os.path.join(args.out_dir, hash_short + '.params'))
    logging.info('param saved to %s. hash = %s', gluon_param_path, hash_full)
    bert.save_parameters(gluon_param_path)
    mx.nd.waitall()
    # save config
    print('finish save vocab')
    print('save model done!'.center(60, '='))


if __name__ == "__main__":
    state_dict = extract_weights(args)
    save_model(state_dict, args.out_dir)