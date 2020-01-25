# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import json
import logging
import os
import sys

import mxnet as mx
import numpy as np

import gluonnlp as nlp
from utils import _split_dict, get_hash, read_tf_checkpoint


def set_params(model, tf_tensors, kwargs, tie_r):
    # Drop optimizer params
    _, tf_tensors = _split_dict(lambda k, v: k.endswith('Adam'), tf_tensors)
    _, tf_tensors = _split_dict(lambda k, v: k.endswith('Adam_1'), tf_tensors)
    del tf_tensors['global_step']

    # Embedding
    tf_param = tf_tensors.pop('model/transformer/word_embedding/lookup_table')
    model._net.word_embed.weight.set_data(mx.nd.array(tf_param))
    tf_param = tf_tensors.pop('model/transformer/mask_emb/mask_emb')
    model._net.mask_embed.set_data(mx.nd.array(tf_param))

    tf_rel_segment_emb = tf_tensors.pop('model/transformer/seg_embed')

    tf_r_r_bias = tf_tensors.pop('model/transformer/r_r_bias')
    tf_r_w_bias = tf_tensors.pop('model/transformer/r_w_bias')
    tf_r_s_bias = tf_tensors.pop('model/transformer/r_s_bias')
    for layer_i in range(kwargs['num_layers']):
        # Attention Cell
        attention_cell = model._net.transformer_cells[layer_i].attention_cell
        # TODO(leezu): Duplicate tied parameters until parameter sharing
        # support is improved in Gluon 2. (It is currently impossible to share
        # only subsets of parameters between Blocks due to name clashes between
        # the non-shared parameters (due to same prefix))
        attention_cell.query_key_bias.set_data(
            mx.nd.array(tf_r_w_bias if tie_r else tf_r_w_bias[layer_i]))
        attention_cell.query_emb_bias.set_data(
            mx.nd.array(tf_r_r_bias if tie_r else tf_r_r_bias[layer_i]))
        attention_cell.query_seg_bias.set_data(
            mx.nd.array(tf_r_s_bias if tie_r else tf_r_s_bias[layer_i]))
        shape = (kwargs['units'], kwargs['units'])
        tf_param = tf_tensors.pop('model/transformer/layer_{}/rel_attn/q/kernel'.format(layer_i))
        attention_cell.proj_query.weight.set_data(mx.nd.array(tf_param.reshape(shape).T))
        tf_param = tf_tensors.pop('model/transformer/layer_{}/rel_attn/k/kernel'.format(layer_i))
        attention_cell.proj_key.weight.set_data(mx.nd.array(tf_param.reshape(shape).T))
        tf_param = tf_tensors.pop('model/transformer/layer_{}/rel_attn/v/kernel'.format(layer_i))
        attention_cell.proj_value.weight.set_data(mx.nd.array(tf_param.reshape(shape).T))
        tf_param = tf_tensors.pop('model/transformer/layer_{}/rel_attn/r/kernel'.format(layer_i))
        attention_cell.proj_emb.weight.set_data(mx.nd.array(tf_param.reshape(shape).T))
        attention_cell.seg_emb.set_data(mx.nd.array(tf_rel_segment_emb[layer_i]))

        # Projection
        tf_param = tf_tensors.pop('model/transformer/layer_{}/rel_attn/o/kernel'.format(layer_i))
        model._net.transformer_cells[layer_i].proj.weight.set_data(
            mx.nd.array(tf_param.reshape(shape)))  # o kernel should not be transposed

        # Layer Norm
        tf_param = tf_tensors.pop(
            'model/transformer/layer_{}/rel_attn/LayerNorm/beta'.format(layer_i))
        model._net.transformer_cells[layer_i].layer_norm.beta.set_data(mx.nd.array(tf_param))
        tf_param = tf_tensors.pop(
            'model/transformer/layer_{}/rel_attn/LayerNorm/gamma'.format(layer_i))
        model._net.transformer_cells[layer_i].layer_norm.gamma.set_data(mx.nd.array(tf_param))

        # FFN
        ffn = model._net.transformer_cells[layer_i].ffn
        tf_param = tf_tensors.pop('model/transformer/layer_{}/ff/LayerNorm/beta'.format(layer_i))
        ffn.layer_norm.beta.set_data(mx.nd.array(tf_param))
        tf_param = tf_tensors.pop('model/transformer/layer_{}/ff/LayerNorm/gamma'.format(layer_i))
        ffn.layer_norm.gamma.set_data(mx.nd.array(tf_param))
        tf_param = tf_tensors.pop('model/transformer/layer_{}/ff/layer_1/kernel'.format(layer_i))
        ffn.ffn_1.weight.set_data(mx.nd.array(tf_param.T))
        tf_param = tf_tensors.pop('model/transformer/layer_{}/ff/layer_1/bias'.format(layer_i))
        ffn.ffn_1.bias.set_data(mx.nd.array(tf_param))
        tf_param = tf_tensors.pop('model/transformer/layer_{}/ff/layer_2/kernel'.format(layer_i))
        ffn.ffn_2.weight.set_data(mx.nd.array(tf_param.T))
        tf_param = tf_tensors.pop('model/transformer/layer_{}/ff/layer_2/bias'.format(layer_i))
        ffn.ffn_2.bias.set_data(mx.nd.array(tf_param))

    if 'model/lm_loss/weight' in tf_tensors:
        tf_param = tf_tensors.pop('model/lm_loss/weight')
        model._net.decoder.weight.set_data(tf_param)
    tf_param = tf_tensors.pop('model/lm_loss/bias')
    model._net.decoder.bias.set_data(tf_param)

    assert len(tf_tensors.keys()) == 0


def convert_xlnet(args):
    # Load vocab
    vocab_file = os.path.join(args.model_dir, 'spiece.model')
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file, cls_token='<cls>', sep_token='<sep>',
                                                   mask_token='<mask>')

    # Load config
    tf_config_names_to_gluon_config_names = {
        'd_inner': 'hidden_size',
        'd_model': 'units',
        'ff_activation': 'activation',
        'n_head': 'num_heads',
        'n_layer': 'num_layers',
        'n_token': 'vocab_size',
    }
    with open(os.path.join(args.model_dir, 'xlnet_config.json'), 'r') as f:
        tf_config = json.load(f)
        assert tf_config['untie_r']
        del tf_config['untie_r']
        del tf_config['d_head']
        assert len(tf_config) == len(tf_config_names_to_gluon_config_names)
    kwargs = {tf_config_names_to_gluon_config_names[k]: v for k, v in tf_config.items()}
    assert len(vocab) == kwargs['vocab_size']
    print(kwargs)

    # Load TF model
    tf_checkpoint_file = os.path.expanduser(os.path.join(args.model_dir, 'xlnet_model.ckpt'))
    tf_tensors = read_tf_checkpoint(tf_checkpoint_file)

    # Update kwargs
    kwargs['tie_decoder_weight'] = 'model/lm_loss/weight' not in tf_tensors

    # Initialize Gluon model
    model = XLNet(**kwargs)
    model.initialize(init=mx.init.Normal(0.02))
    model.hybridize()

    # Shape inference based on forward pass
    batch_size, qlen, mlen = 2, 16, 100
    mems = model.begin_mems(batch_size, mlen, context=mx.cpu())
    x = mx.nd.ones(shape=(batch_size, qlen))
    segments = mx.nd.random_normal(shape=(batch_size, qlen, mlen + qlen, 2))
    segments = segments < 0
    model(x, segments, mems)

    # Convert parameters
    set_params(model, tf_tensors, kwargs, tie_r=False)

    # Serialization
    tmp_file_path = os.path.expanduser(os.path.join(args.out_dir, 'tmp'))
    with open(tmp_file_path, 'w') as f:
        f.write(vocab.to_json())
    hash_full, hash_short = get_hash(tmp_file_path)
    gluon_vocab_path = os.path.expanduser(os.path.join(args.out_dir, hash_short + '.vocab'))
    with open(gluon_vocab_path, 'w') as f:
        f.write(vocab.to_json())
        logging.info('vocab file saved to %s. hash = %s', gluon_vocab_path, hash_full)
    model.save_parameters(tmp_file_path)
    hash_full, hash_short = get_hash(tmp_file_path)
    os.remove(tmp_file_path)
    gluon_param_path = os.path.expanduser(os.path.join(args.out_dir, hash_short + '.params'))
    logging.info('param saved to %s. hash = %s', gluon_param_path, hash_full)
    model.save_parameters(gluon_param_path)
    mx.nd.waitall()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conversion script for Tensorflow XLNet model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model-dir', type=str, required=True,
        help='Path to folder including the TensorFlow checkpoint `xlnet_model.ckpt`, '
        'the SentencePiece model `spiece.model` and the modle config `xlnet_config.json`')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Path to output folder. The folder must exist.')
    parser.add_argument('--debug', action='store_true', help='debugging mode')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.info(args)

    sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
    from transformer import XLNet

    convert_xlnet(args)
