import os
import re
import json
import sys
import shutil
import logging
import argparse

import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose

from gluonnlp.utils.misc import sha1sum, naming_convention, logging_config
from gluonnlp.data.tokenizers import HuggingFaceWordPieceTokenizer
from gluonnlp.models.mobilebert import MobileBertModel, MobileBertForPretrain
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

mx.npx.set_np()
np.random.seed(1234)
mx.npx.random.seed(1234)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the TF Mobile Bert Model to Gluon')
    parser.add_argument('--tf_model_path', type=str,
                        help='Directory of the model downloaded from TF hub.')
    parser.add_argument('--mobilebert_dir', type=str,
                        help='Path to the github repository of electra, you may clone it by '
                             '`svn checkout https://github.com/google-research/google-research/trunk/mobilebert`.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='directory path to save the converted Mobile Bert model.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='a single gpu to run mxnet, e.g. 0 or 1 The default device is cpu ')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    return args


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


def convert_tf_config(config_dict_path, vocab_size):
    """Convert the config file"""
    with open(config_dict_path, encoding='utf-8') as f:
        config_dict = json.load(f)
    assert vocab_size == config_dict['vocab_size']
    cfg = MobileBertModel.get_cfg().clone()
    cfg.defrost()
    cfg.MODEL.vocab_size = vocab_size
    cfg.MODEL.units = config_dict['hidden_size']
    cfg.MODEL.embed_size = config_dict['embedding_size']
    cfg.MODEL.inner_size = config_dict['intra_bottleneck_size']
    cfg.MODEL.hidden_size = config_dict['intermediate_size']
    cfg.MODEL.max_length = config_dict['max_position_embeddings']
    cfg.MODEL.num_heads = config_dict['num_attention_heads']
    cfg.MODEL.num_layers = config_dict['num_hidden_layers']
    cfg.MODEL.bottleneck_strategy
    cfg.MODEL.num_stacked_ffn = config_dict['num_feedforward_networks']
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.activation = config_dict['hidden_act']
    cfg.MODEL.num_token_types = config_dict['type_vocab_size']
    cfg.MODEL.hidden_dropout_prob = float(config_dict['hidden_dropout_prob'])
    cfg.MODEL.attention_dropout_prob = float(config_dict['attention_probs_dropout_prob'])
    cfg.MODEL.normalization = config_dict['normalization_type']
    cfg.MODEL.dtype = 'float32'

    if 'use_bottleneck_attention' in config_dict.keys():
        cfg.MODEL.bottleneck_strategy = 'from_bottleneck'
    elif 'key_query_shared_bottleneck' in config_dict.keys():
        cfg.MODEL.bottleneck_strategy = 'qk_sharing'
    else:
        cfg.MODEL.bottleneck_strategy = 'from_input'

    cfg.INITIALIZER.weight = ['truncnorm', 0,
                              config_dict['initializer_range']]  # TruncNorm(0, 0.02)
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


def convert_tf_assets(tf_assets_dir):
    """Convert the assets file including config, vocab and tokenizer model"""
    file_names = os.listdir(tf_assets_dir)
    vocab_path = None
    json_cfg_path = None
    for ele in file_names:
        if ele.endswith('.txt'):
            assert vocab_path is None
            vocab_path = ele
        elif ele.endswith('.json'):
            assert json_cfg_path is None
            json_cfg_path = ele
    assert vocab_path is not None and json_cfg_path is not None

    vocab_path = os.path.join(tf_assets_dir, vocab_path)
    vocab_size = len(open(vocab_path, 'r', encoding='utf-8').readlines())
    json_cfg_path = os.path.join(tf_assets_dir, json_cfg_path)
    cfg = convert_tf_config(json_cfg_path, vocab_size)
    return cfg, json_cfg_path, vocab_path


CONVERT_MAP = [
    # mlm model
    ('cls/', ''),
    ('predictions/extra_output_weights', 'extra_table.weight'),
    ('predictions/output_bias', 'embedding_table.bias'),
    ('predictions/transform/LayerNorm', 'mlm_decoder.2'),
    ('predictions/transform/dense', 'mlm_decoder.0'),
    ('seq_relationship/output_bias', 'nsp_classifier.bias'),
    ('seq_relationship/output_weights', 'nsp_classifier.weight'),
    # backbone
    ('bert/', 'backbone_model.'),
    ('layer_', 'all_layers.'),
    ('attention/output/FakeLayerNorm', 'layer_norm'),
    ('attention/output/dense', 'attention_proj'),
    # inner ffn layer denoted by xxx
    ('ffn_layers_xxx/intermediate/dense', 'stacked_ffn.xxx.ffn_1'),
    ('ffn_layers_xxx/output/FakeLayerNorm', 'stacked_ffn.xxx.layer_norm'),
    ('ffn_layers_xxx/output/dense', 'stacked_ffn.xxx.ffn_2'),
    # last ffn layer denoted by xxy
    ('intermediate/dense', 'stacked_ffn.xxy.ffn_1'),
    ('output/FakeLayerNorm', 'stacked_ffn.xxy.layer_norm'),
    ('output/dense', 'stacked_ffn.xxy.ffn_2'),
    # embeddings
    ('embeddings/word_embeddings', 'word_embed.weight'),
    ('embeddings/token_type_embeddings', 'token_type_embed.weight'),
    ('embeddings/position_embeddings', 'token_pos_embed._embed.weight'),
    ('embeddings/embedding_transformation', 'embed_factorized_proj'),
    ('embeddings/FakeLayerNorm', 'embed_layer_norm'),
    ('bottleneck/input/FakeLayerNorm', 'in_bottleneck_ln'),
    ('bottleneck/input/dense', 'in_bottleneck_proj'),
    ('bottleneck/attention/FakeLayerNorm', 'shared_qk_ln'),
    ('bottleneck/attention/dense', 'shared_qk'),
    ('output/bottleneck/FakeLayerNorm', 'out_bottleneck_ln'),
    ('output/bottleneck/dense', 'out_bottleneck_proj'),
    ('attention/self/key', 'attn_key'),
    ('attention/self/query', 'attn_query'),
    ('attention/self/value', 'attn_value'),
    ('output/', ''),
    ('kernel', 'weight'),
    ('FakeLayerNorm', 'layer_norm'),
    ('LayerNorm', 'layer_norm'),
    ('/', '.'),
]


def get_name_map(tf_names, num_stacked_ffn):
    """
    Get the converting mapping between tensor names and mxnet names.
    The above mapping CONVERT_MAP is effectively adaptive to Bert and Albert,
    but there is no guarantee that it can match to other tf models in case of
    some sepecial variable_scope (tensorflow) and prefix (mxnet).

    Redefined mapping is encouraged to adapt the personalization model.

    Parameters
    ----------
    tf_names
        the parameters names of tensorflow model
    Returns
    -------
    A dictionary with the following format:
        {tf_names : mx_names}
    """
    name_map = {}
    for source_name in tf_names:
        target_name = source_name
        ffn_idx = re.findall(r'ffn_layer_\d+', target_name)
        if ffn_idx:
            target_name = target_name.replace(ffn_idx[0], 'ffn_layers_xxx')
        for old, new in CONVERT_MAP:
            target_name = target_name.replace(old, new)
        if ffn_idx:
            target_name = target_name.replace('stacked_ffn.xxx', 'stacked_ffn.' + ffn_idx[0][10:])
        if 'stacked_ffn.xxy' in target_name:
            target_name = target_name.replace(
                'stacked_ffn.xxy', 'stacked_ffn.' + str(num_stacked_ffn - 1))
        name_map[source_name] = target_name

    return name_map


def convert_tf_model(model_dir, save_dir, test_conversion, gpu, mobilebert_dir):
    ctx = mx.gpu(gpu) if gpu is not None else mx.cpu()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cfg, json_cfg_path, vocab_path = convert_tf_assets(model_dir)
    with open(os.path.join(save_dir, 'model.yml'), 'w') as of:
        of.write(cfg.dump())
    new_vocab = HuggingFaceWordPieceTokenizer(
        vocab_file=vocab_path,
        unk_token='[UNK]',
        pad_token='[PAD]',
        cls_token='[CLS]',
        sep_token='[SEP]',
        mask_token='[MASK]',
        lowercase=True).vocab
    new_vocab.save(os.path.join(save_dir, 'vocab.json'))

    # test input data
    batch_size = 3
    seq_length = 32
    num_mask = 5
    input_ids = np.random.randint(0, cfg.MODEL.vocab_size, (batch_size, seq_length))
    valid_length = np.random.randint(seq_length // 2, seq_length, (batch_size,))
    input_mask = np.broadcast_to(np.arange(seq_length).reshape(1, -1), (batch_size, seq_length)) \
        < np.expand_dims(valid_length, 1)
    segment_ids = np.random.randint(0, 2, (batch_size, seq_length))
    mlm_positions = np.random.randint(0, seq_length // 2, (batch_size, num_mask))

    tf_input_ids = tf.constant(input_ids, dtype=np.int32)
    tf_input_mask = tf.constant(input_mask, dtype=np.int32)
    tf_segment_ids = tf.constant(segment_ids, dtype=np.int32)

    init_checkpoint = os.path.join(model_dir, 'mobilebert_variables.ckpt')
    tf_params = read_tf_checkpoint(init_checkpoint)
    # get parameter names for tensorflow with unused parameters filtered out.
    tf_names = sorted(tf_params.keys())
    tf_names = filter(lambda name: not name.endswith('adam_m'), tf_names)
    tf_names = filter(lambda name: not name.endswith('adam_v'), tf_names)
    tf_names = filter(lambda name: name != 'global_step', tf_names)
    tf_names = list(tf_names)

    sys.path.append(mobilebert_dir)
    from mobilebert import modeling

    tf_bert_config = modeling.BertConfig.from_json_file(json_cfg_path)
    bert_model = modeling.BertModel(
        config=tf_bert_config,
        is_training=False,
        input_ids=tf_input_ids,
        input_mask=tf_input_mask,
        token_type_ids=tf_segment_ids,
        use_one_hot_embeddings=False)
    tvars = tf.trainable_variables()
    assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # the name of the parameters are ending with ':0' like 'Mobile
        # Bert/embeddings/word_embeddings:0'
        backbone_params = {v.name.split(":")[0]: v.read_value() for v in tvars}
        backbone_params = sess.run(backbone_params)
        tf_token_outputs_np = {
            'pooled_output': sess.run(bert_model.get_pooled_output()),
            'sequence_output': sess.run(bert_model.get_sequence_output()),
        }

    # The following part only ensure the parameters in backbone model are valid
    for k in backbone_params:
        assert_allclose(tf_params[k], backbone_params[k])

    # Build gluon model and initialize
    gluon_pretrain_model = MobileBertForPretrain(cfg)
    gluon_pretrain_model.initialize(ctx=ctx)
    gluon_pretrain_model.hybridize()

    # pepare test data
    mx_input_ids = mx.np.array(input_ids, dtype=np.int32, ctx=ctx)
    mx_valid_length = mx.np.array(valid_length, dtype=np.int32, ctx=ctx)
    mx_token_types = mx.np.array(segment_ids, dtype=np.int32, ctx=ctx)
    mx_masked_positions = mx.np.array(mlm_positions, dtype=np.int32, ctx=ctx)

    has_mlm = True
    name_map = get_name_map(tf_names, cfg.MODEL.num_stacked_ffn)
    # go through the gluon model to infer the shape of parameters
    model = gluon_pretrain_model
    contextual_embedding, pooled_output, nsp_score, mlm_scores = \
        model(mx_input_ids, mx_token_types, mx_valid_length, mx_masked_positions)
    # replace tensorflow parameter names with gluon parameter names
    mx_params = model.collect_params()
    all_keys = set(mx_params.keys())
    for (src_name, dst_name) in name_map.items():
        tf_param_val = tf_params[src_name]
        if dst_name is None:
            continue
        all_keys.remove(dst_name)
        if src_name.endswith('kernel'):
            mx_params[dst_name].set_data(tf_param_val.T)
        else:
            mx_params[dst_name].set_data(tf_param_val)

    if has_mlm:
        # 'embedding_table.weight' is shared with word_embed.weight
        all_keys.remove('embedding_table.weight')
    assert len(all_keys) == 0, 'parameters missing from tensorflow checkpoint'

    # test conversion results for backbone model
    if test_conversion:
        tf_contextual_embedding = tf_token_outputs_np['sequence_output']
        tf_pooled_output = tf_token_outputs_np['pooled_output']
        contextual_embedding, pooled_output = model.backbone_model(
            mx_input_ids, mx_token_types, mx_valid_length)
        assert_allclose(pooled_output.asnumpy(), tf_pooled_output, 1E-2, 1E-2)
        for i in range(batch_size):
            ele_valid_length = valid_length[i]
            assert_allclose(contextual_embedding[i, :ele_valid_length, :].asnumpy(),
                            tf_contextual_embedding[i, :ele_valid_length, :], 1E-2, 1E-2)
    model.backbone_model.save_parameters(os.path.join(save_dir, 'model.params'), deduplicate=True)
    logging.info('Convert the backbone model in {} to {}/{}'.format(model_dir, save_dir, 'model.params'))
    model.save_parameters(os.path.join(save_dir, 'model_mlm.params'), deduplicate=True)
    logging.info('Convert the MLM and NSP model in {} to {}/{}'.format(model_dir,
                                                                       save_dir, 'model_mlm.params'))

    logging.info('Conversion finished!')
    logging.info('Statistics:')

    old_names = os.listdir(save_dir)
    for old_name in old_names:
        new_name, long_hash = naming_convention(save_dir, old_name)
        old_path = os.path.join(save_dir, old_name)
        new_path = os.path.join(save_dir, new_name)
        shutil.move(old_path, new_path)
        file_size = os.path.getsize(new_path)
        logging.info('\t{}/{} {} {}'.format(save_dir, new_name, long_hash, file_size))


if __name__ == '__main__':
    args = parse_args()
    logging_config()
    save_dir = args.save_dir if args.save_dir is not None else os.path.basename(
        args.tf_model_path) + '_gluon'
    mobilebert_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(
                args.mobilebert_dir),
            os.path.pardir))
    convert_tf_model(args.tf_model_path, save_dir, args.test, args.gpu, mobilebert_dir)
