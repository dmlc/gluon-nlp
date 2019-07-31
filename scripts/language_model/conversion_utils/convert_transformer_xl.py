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
import logging
import os
import pickle
import re
import sys

import mxnet as mx
import numpy as np

import gluonnlp as nlp
from utils import _split_dict, get_hash, to_gluon_kwargs, read_tf_checkpoint


def to_gluon_vocab(corpus):
    """Convert a TransformerXL corpus object to a GluonNLP Vocab."""
    # Clean up latin-1 mis-encoding of words
    idx2sym = [w.encode('latin-1').decode('utf-8') for w in corpus.vocab.idx2sym]
    sym2idx = {sym: idx for idx, sym in enumerate(idx2sym)}

    special_tokens = dict(unknown_token=None, padding_token=None, bos_token=None)
    if hasattr(corpus.vocab, 'unk_idx'):
        special_tokens['unknown_token'] = idx2sym[corpus.vocab.unk_idx]
    elif '<unk>' in sym2idx:
        special_tokens['unknown_token'] = '<unk>'
    elif '<UNK>' in sym2idx:
        special_tokens['unknown_token'] = '<UNK>'

    # Discover special tokens
    if ['<eos>'] == corpus.vocab.special:
        if '<eos>' in sym2idx:  # Only include if special token is actually used
            special_tokens['eos_token'] = '<eos>'
    elif '<S>' in sym2idx:
        # Special case for model trained on Google 1 Billion Word LM dataset
        special_tokens['eos_token'] = '<S>'
    elif corpus.vocab.special:
        raise NotImplementedError('Provided TransformerXL cache.pkl uses an unknown special token. '
                                  'You must extend the `to_gluon_vocab` method to support it.')
    else:
        special_tokens['eos_token'] = None

    counter = nlp.data.count_tokens(sym2idx.keys())
    vocab = nlp.vocab.Vocab(counter, token_to_idx=sym2idx, **special_tokens)
    return vocab


def set_params(model, tf_tensors, kwargs, tie_r):
    # Drop optimizer params
    _, tf_tensors = _split_dict(lambda k, v: k.endswith('Adam'), tf_tensors)
    _, tf_tensors = _split_dict(lambda k, v: k.endswith('Adam_1'), tf_tensors)
    del tf_tensors['global_step']
    del tf_tensors['beta1_power']
    del tf_tensors['beta2_power']

    loaded = set()  # Cache of processed parameters

    if 'embed_cutoffs' in kwargs:  # Adaptive Embedding and Softmax
        # Embedding
        for name, param in model._net.embedding._collect_params_with_prefix().items():
            purpose, i, postfix = re.match(r'([a-zA-Z]*)(\d*)(.*)', name).groups()
            if purpose == 'embedding':
                assert postfix == '_weight'
                tf_param = tf_tensors.pop(
                    'transformer/adaptive_embed/cutoff_{}/lookup_table'.format(i))
            elif purpose == 'projection':
                assert postfix == '_weight'
                tf_param = tf_tensors.pop('transformer/adaptive_embed/cutoff_{}/proj_W'.format(i)).T
            else:
                raise RuntimeError('Embedding had unexpected parameter: {}'.format(name))

            param.set_data(mx.nd.array(tf_param))
            loaded.add(param)

        # Softmax
        for name, param in model._net.crit._collect_params_with_prefix().items():
            if param in loaded:
                continue  # Some parameters are shared between Embedding and Softmax

            purpose, i, postfix = re.match(r'([a-zA-Z]*)(\d*)(.*)', name).groups()
            if purpose == 'outembedding':
                if postfix == '_weight':
                    tf_param = tf_tensors.pop(
                        'transformer/adaptive_softmax/cutoff_{}/lookup_table'.format(i))
                elif postfix == '_bias':
                    tf_param = tf_tensors.pop('transformer/adaptive_softmax/cutoff_{}/b'.format(i))
                else:
                    raise RuntimeError('Softmax had unexpected parameter: {}'.format(name))
            elif purpose == 'outprojection':
                assert postfix == '_weight'
                tf_param = tf_tensors.pop('transformer/adaptive_softmax/cutoff_{}/proj'.format(i)).T
            elif purpose == 'cluster':
                if postfix == '.weight':
                    tf_param = tf_tensors.pop('transformer/adaptive_softmax/cutoff_0/cluster_W')
                elif postfix == '.bias':
                    tf_param = tf_tensors.pop('transformer/adaptive_softmax/cutoff_0/cluster_b')
                else:
                    raise RuntimeError('Softmax had unexpected parameter: {}'.format(name))
            else:
                raise RuntimeError('Softmax had unexpected parameter: {}'.format(name))

            param.set_data(mx.nd.array(tf_param))
            loaded.add(param)
    else:  # Non-adaptive, (possibly) projected embedding and softmax
        # Embedding
        tf_param = tf_tensors.pop('transformer/adaptive_embed/lookup_table')
        model._net.embedding.embedding_weight.set_data(mx.nd.array(tf_param))
        loaded.add(model._net.embedding.embedding_weight)
        if kwargs['embed_size'] != kwargs['units']:
            tf_param = tf_tensors.pop('transformer/adaptive_embed/proj_W')
            model._net.embedding.projection_weight.set_data(mx.nd.array(tf_param))
            loaded.add(model._net.embedding.projection_weight)
            assert len(model._net.embedding.collect_params().keys()) == 2
        else:
            assert len(model._net.embedding.collect_params().keys()) == 1

        # Softmax
        for name, param in model._net.crit._collect_params_with_prefix().items():
            if param in loaded:
                continue  # Some parameters are shared between Embedding and Softmax

            purpose, i, postfix = re.match(r'([a-zA-Z]*)(\d*)(.*)', name).groups()
            if purpose == 'outembedding':
                if postfix == '_weight':
                    tf_param = tf_tensors.pop('transformer/adaptive_softmax/lookup_table')
                elif postfix == '_bias':
                    tf_param = tf_tensors.pop('transformer/adaptive_softmax/bias')
                else:
                    raise RuntimeError('Softmax had unexpected parameter: {}'.format(name))
            elif purpose == 'outprojection':
                assert postfix == '_weight'
                tf_param = tf_tensors.pop('transformer/adaptive_softmax/proj').T
            else:
                raise RuntimeError('Softmax had unexpected parameter: {}'.format(name))

            param.set_data(mx.nd.array(tf_param))
            loaded.add(param)

    tf_r_r_bias = tf_tensors.pop('transformer/r_r_bias')
    tf_r_w_bias = tf_tensors.pop('transformer/r_w_bias')
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
        tf_param = np.split(
            tf_tensors.pop('transformer/layer_{}/rel_attn/qkv/kernel'.format(layer_i)).T, 3, axis=0)
        attention_cell.proj_query.weight.set_data(mx.nd.array(tf_param[0]))
        attention_cell.proj_key.weight.set_data(mx.nd.array(tf_param[1]))
        attention_cell.proj_value.weight.set_data(mx.nd.array(tf_param[2]))
        tf_param = tf_tensors.pop('transformer/layer_{}/rel_attn/r/kernel'.format(layer_i))
        attention_cell.proj_emb.weight.set_data(mx.nd.array(tf_param.T))

        # Projection
        tf_param = tf_tensors.pop('transformer/layer_{}/rel_attn/o/kernel'.format(layer_i))
        model._net.transformer_cells[layer_i].proj.weight.set_data(mx.nd.array(tf_param.T))

        # Layer Norm
        tf_param = tf_tensors.pop('transformer/layer_{}/rel_attn/LayerNorm/beta'.format(layer_i))
        model._net.transformer_cells[layer_i].layer_norm.beta.set_data(mx.nd.array(tf_param))
        tf_param = tf_tensors.pop('transformer/layer_{}/rel_attn/LayerNorm/gamma'.format(layer_i))
        model._net.transformer_cells[layer_i].layer_norm.gamma.set_data(mx.nd.array(tf_param))

        # FFN
        ffn = model._net.transformer_cells[layer_i].ffn
        tf_param = tf_tensors.pop('transformer/layer_{}/ff/LayerNorm/beta'.format(layer_i))
        ffn.layer_norm.beta.set_data(mx.nd.array(tf_param))
        tf_param = tf_tensors.pop('transformer/layer_{}/ff/LayerNorm/gamma'.format(layer_i))
        ffn.layer_norm.gamma.set_data(mx.nd.array(tf_param))
        tf_param = tf_tensors.pop('transformer/layer_{}/ff/layer_1/kernel'.format(layer_i))
        ffn.ffn_1.weight.set_data(mx.nd.array(tf_param.T))
        tf_param = tf_tensors.pop('transformer/layer_{}/ff/layer_1/bias'.format(layer_i))
        ffn.ffn_1.bias.set_data(mx.nd.array(tf_param))
        tf_param = tf_tensors.pop('transformer/layer_{}/ff/layer_2/kernel'.format(layer_i))
        ffn.ffn_2.weight.set_data(mx.nd.array(tf_param.T))
        tf_param = tf_tensors.pop('transformer/layer_{}/ff/layer_2/bias'.format(layer_i))
        ffn.ffn_2.bias.set_data(mx.nd.array(tf_param))


def convert_transformerxl(args):
    # Load tf model and vocab
    with open(args.cache_pkl, 'rb') as f:
        corpus = pickle.load(f, encoding='latin1')
    vocab = to_gluon_vocab(corpus)
    tf_checkpoint_file = os.path.expanduser(
        os.path.join(args.tf_checkpoint_dir, args.tf_model_prefix))
    tf_tensors = read_tf_checkpoint(tf_checkpoint_file)

    # Initialize Gluon model
    kwargs, tie_r = to_gluon_kwargs(tf_tensors)
    model = TransformerXL(vocab_size=len(vocab), **kwargs)
    model.initialize(init=mx.init.Normal(0.02))

    # Shape inference based on forward pass
    batch_size, seq_len = 2, 16
    mem_length = 100
    mems = model.begin_mems(batch_size, mem_length, context=mx.cpu())
    x = mx.nd.ones(shape=(batch_size, seq_len))
    model(x, x, mems)

    # Convert parameters
    set_params(model, tf_tensors, kwargs, tie_r)

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
    parser = argparse.ArgumentParser(
        description='Conversion script for Tensorflow Transformer-XL model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--transformer-xl-repo', type=str, required=True,
                        help='Path to https://github.com/kimiyoung/transformer-xl repo.')
    parser.add_argument('--tf-checkpoint-dir', type=str, required=True,
                        help='Path to Tensorflow checkpoint folder.')
    parser.add_argument(
        '--tf-model-prefix', type=str, required=True, help='Prefix of the checkpoint files. '
        'For example model.ckpt-0 or model.ckpt-1191000')
    parser.add_argument('--cache-pkl', type=str, required=True,
                        help='Path to TransformerXL cache.pkl file.')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Path to output folder. The folder must exist.')
    parser.add_argument('--debug', action='store_true', help='debugging mode')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.info(args)

    # Load stuff required for unpickling
    sys.path.append(os.path.join((args.transformer_xl_repo), 'tf'))
    import vocabulary  # pylint: disable=unused-import
    import data_utils  # pylint: disable=unused-import

    sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
    from transformer import TransformerXL

    convert_transformerxl(args)
