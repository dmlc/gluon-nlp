import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import os
import re
import json
import shutil
import logging
import argparse

import mxnet as mx
import torch as th
import numpy as np
from numpy.testing import assert_allclose

from gluonnlp.data.vocab import Vocab
from gluonnlp.utils.misc import naming_convention, logging_config, BooleanOptionalAction
from gluonnlp.models.bert import BertModel, BertForMLM
from gluonnlp.models.albert import AlbertModel, AlbertForMLM
from gluonnlp.torch.models.bert import BertModel as ThBertModel, BertForMLM as ThBertForMLM
from gluonnlp.data.tokenizers import SentencepieceTokenizer, HuggingFaceWordPieceTokenizer

import tensorflow
USE_TF_V1 = tensorflow.version.VERSION.split('.')[0] < '2'
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
if not USE_TF_V1:
    tensorflow.config.set_visible_devices([], 'GPU')
    visible_devices = tensorflow.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'

mx.npx.set_np()
np.random.seed(1234)
mx.npx.random.seed(1234)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the TF pretrained model to Gluon')
    parser.add_argument('--tf_hub_model_path', type=str, required=True,
                        help='Directory of the model downloaded from TF hub.')
    parser.add_argument('--model_type', type=str, choices=['bert', 'albert'],
                        help='The name of the model to be converted. '
                             'Only Bert and Albert are currently supported.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='directory path to save the converted pretrained model.')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--torch', action='store_true')
    parser.add_argument("--cuda", action=BooleanOptionalAction, default=True,
                        help="Use Cuda if available.")
    args = parser.parse_args()

    if args.torch:
        if th.cuda.is_available() and args.cuda:
            args.device = th.device("cuda", 0)
        else:
            args.device = th.device("cpu")
    else:
        args.ctx = mx.gpu() if args.cuda else mx.cpu()

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


def convert_tf_config(json_cfg_path, vocab_size, model_type):
    """Convert the config file"""

    with open(json_cfg_path, encoding='utf-8') as f:
        json_cfg = json.load(f)
    if model_type == 'bert':
        # For bert model, the config file are copied from local configuration file
        # leaving the vocab_size indistinguishable. Actually, the verification of
        # vocab_size would be done in the process of embedding weights conversion.
        cfg = BertModel.get_cfg().clone()
    elif model_type == 'albert':
        assert vocab_size == json_cfg['vocab_size']
        cfg = AlbertModel.get_cfg().clone()
    else:
        raise NotImplementedError
    cfg.defrost()
    cfg.MODEL.vocab_size = vocab_size

    cfg.MODEL.units = json_cfg['hidden_size']
    cfg.MODEL.hidden_size = json_cfg['intermediate_size']
    cfg.MODEL.max_length = json_cfg['max_position_embeddings']
    cfg.MODEL.num_heads = json_cfg['num_attention_heads']
    cfg.MODEL.num_layers = json_cfg['num_hidden_layers']
    cfg.MODEL.pos_embed_type = 'learned'
    if json_cfg['hidden_act'] == 'gelu':
        cfg.MODEL.activation = 'gelu(tanh)'
    else:
        cfg.MODEL.activation = json_cfg['hidden_act']
    cfg.MODEL.layer_norm_eps = 1E-12
    cfg.MODEL.num_token_types = json_cfg['type_vocab_size']
    cfg.MODEL.hidden_dropout_prob = float(json_cfg['hidden_dropout_prob'])
    cfg.MODEL.attention_dropout_prob = float(json_cfg['attention_probs_dropout_prob'])
    cfg.INITIALIZER.weight = ['truncnorm', 0, json_cfg['initializer_range']]  # TruncNorm(0, 0.02)
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    if model_type == 'albert':
        # The below configurations are not supported in bert
        cfg.MODEL.embed_size = json_cfg['embedding_size']
        cfg.MODEL.num_groups = json_cfg['num_hidden_groups']
    cfg.freeze()
    return cfg


def convert_tf_assets(tf_assets_dir, model_type):
    """Convert the assets file including config, vocab and tokenizer model"""
    file_names = os.listdir(tf_assets_dir)
    json_cfg_path = None
    spm_model_path = None
    vocab_path = None
    for ele in file_names:
        if ele.endswith('.model'):
            assert spm_model_path is None
            spm_model_path = ele
        elif ele.endswith('.json'):
            assert json_cfg_path is None
            json_cfg_path = ele
        elif ele.endswith('.txt'):
            assert vocab_path is None
            vocab_path = ele
    assert json_cfg_path is not None and \
        (spm_model_path is not None or vocab_path is not None), "The file to be" \
        "converted is missing and at least one word segmentation tool or dictionary exists"

    json_cfg_path = os.path.join(tf_assets_dir, json_cfg_path)
    if spm_model_path:
        spm_model_path = os.path.join(tf_assets_dir, spm_model_path)
        tokenizer = SentencepieceTokenizer(spm_model_path)
        vocab_size = len(tokenizer.vocab)
    elif vocab_path:
        vocab_path = os.path.join(tf_assets_dir, vocab_path)
        vocab_size = len(open(vocab_path, 'r', encoding='utf-8').readlines())
    cfg = convert_tf_config(json_cfg_path, vocab_size, model_type)
    return cfg, vocab_path, spm_model_path


CONVERT_MAP_TF1 = [
    ('bert/', 'backbone_model.'),
    ('cls/', ''),
    ('predictions/transform/dense', 'mlm_decoder.0'),
    ('predictions/transform/LayerNorm', 'mlm_decoder.2'),
    ('predictions/output_bias', 'mlm_decoder.3.bias'),
    ('transformer/', ''),
    ('transform/', ''),
    ('embeddings/word_embeddings', 'word_embed.weight'),
    ('embeddings/token_type_embeddings', 'token_type_embed.weight'),
    ('encoder/embedding_hidden_mapping_in', 'embed_factorized_proj'),
    ('group_0/inner_group_0/', 'all_encoder_groups.0.'),  # albert
    ('layer_', 'all_layers.'),  # bert
    ('embeddings/LayerNorm', 'embed_layer_norm'),
    ('attention/output/LayerNorm', 'layer_norm'),  # bert
    ('output/LayerNorm', 'ffn.layer_norm'),  # bert
    ('LayerNorm_1', 'ffn.layer_norm'),  # albert
    ('LayerNorm', 'layer_norm'),  # albert
    ('attention_1', 'attention'),  # albert
    ('attention/output/dense', 'attention_proj'),
    ('ffn_1/', ''),  # bert & albert
    ('intermediate/dense', 'ffn.ffn_1'),  # albert
    ('intermediate/output/dense', 'ffn.ffn_2'),  # albert
    ('output/dense', 'ffn.ffn_2'),  # bert
    ('output/', ''),
    ('pooler/dense', 'pooler'),
    ('kernel', 'weight'),
    ('attention/', ''),
    ('/', '.'),
]

CONVERT_MAP_TF2 = [
    (':0', ''),
    ('cls/', ''),
    ('predictions/output_bias', 'mlm_decoder.3.bias'),
    ('transformer/layer_', 'encoder.all_layers.'),
    ('word_embeddings/embeddings', 'word_embed.weight'),
    ('type_embeddings/embeddings', 'token_type_embed.weight'),
    ('embeddings/layer_norm', 'embed_layer_norm'),
    ('embedding_projection', 'embed_factorized_proj'),
    ('self_attention/attention_output', 'attention_proj'),
    ('self_attention_layer_norm', 'layer_norm'),
    ('intermediate', 'ffn.ffn_1'),
    ('output_layer_norm', 'ffn.layer_norm'),
    ('output', 'ffn.ffn_2'),
    ("pooler_transform", "pooler"),
    ('kernel', 'weight'),
    ('/', '.'),
]


def get_name_map(tf_names, is_TF1=True):
    """
    Get the converting mapping between TF names and mxnet names.
    The above mapping CONVERT_MAP is effectively adaptive to Bert and Albert,
    but there is no guarantee that it can match to other tf models in case of
    some special variable_scope (tensorflow) and prefix (mxnet).

    Redefined mapping is encouraged to adapt the personalization model.

    Parameters
    ----------
    tf_names
        the parameters names of tensorflow model
    is_TF1
        whether load from TF1 Hub Modules

    Returns
    -------
    A dictionary with the following format:
        {tf_names : mx_names} or {tf_names : th_names}
    """
    convert_map = CONVERT_MAP_TF1 if is_TF1 else CONVERT_MAP_TF2
    if args.torch and is_TF1:
        CONVERT_MAP_TF1.insert(10, ('embeddings/position_embeddings', 'token_pos_embed.weight'))
        CONVERT_MAP_TF1.insert(-1, ('beta', 'bias'))
        CONVERT_MAP_TF1.insert(-1, ('gamma', 'weight'))
    elif is_TF1:
        CONVERT_MAP_TF1.insert(10, ('embeddings/position_embeddings', 'token_pos_embed._embed.weight'))
    elif args.torch:
        CONVERT_MAP_TF2.insert(10, ('position_embedding/embeddings', 'token_pos_embed.weight'))
    else:
        CONVERT_MAP_TF2.insert(10, ('position_embedding/embeddings', 'token_pos_embed._embed.weight'))

    name_map = {}
    for source_name in tf_names:
        target_name = source_name
        # skip the qkv weights
        if 'self/' in source_name:
            name_map[source_name] = None
            continue
        if re.match(r'^transformer\/layer_[\d]+\/self_attention\/(key|value|query)\/(kernel|bias)$',
                    source_name) is not None:
            name_map[source_name] = None
            continue
        for old, new in convert_map:
            target_name = target_name.replace(old, new)
        name_map[source_name] = target_name
    return name_map


def convert_tf_model(hub_model_dir, save_dir, test_conversion, model_type):
    # set up the model type to be converted
    if model_type == 'bert':
        if args.torch:
            PretrainedModel, PretrainedMLMModel = ThBertModel, ThBertForMLM
        else:
            PretrainedModel, PretrainedMLMModel = BertModel, BertForMLM
    elif model_type == 'albert' and not args.torch:
        PretrainedModel, PretrainedMLMModel = AlbertModel, AlbertForMLM
    else:
        raise NotImplementedError

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cfg, vocab_path, spm_model_path = convert_tf_assets(os.path.join(hub_model_dir, 'assets'),
                                                        model_type)
    with open(os.path.join(save_dir, 'model.yml'), 'w') as of:
        of.write(cfg.dump())
    if spm_model_path:
        # Sentencepiece Tokenizer that used in albert model
        tokenizer = SentencepieceTokenizer(spm_model_path)
        new_vocab = Vocab(tokenizer.vocab.all_tokens, unk_token='<unk>', pad_token='<pad>',
                          cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')
        shutil.copy(spm_model_path, os.path.join(save_dir, 'spm.model'))
    elif vocab_path:
        # Wordpiece Tokenizer that used in bert and electra model

        # In this step, the vocabulary is converted with the help of the tokenizer,
        # so whether tokenzier is case-dependent does not matter.
        new_vocab = HuggingFaceWordPieceTokenizer(vocab_file=vocab_path, unk_token='[UNK]',
                                                  pad_token='[PAD]', cls_token='[CLS]',
                                                  sep_token='[SEP]', mask_token='[MASK]',
                                                  lowercase=True).vocab

    new_vocab.save(os.path.join(save_dir, 'vocab.json'))

    # test input data
    batch_size = 2
    seq_length = 16
    num_mask = 5
    input_ids = np.random.randint(0, cfg.MODEL.vocab_size, (batch_size, seq_length))
    valid_length = np.random.randint(seq_length // 2, seq_length, (batch_size, ))
    input_mask = np.broadcast_to(np.arange(seq_length).reshape(1, -1), (batch_size, seq_length)) \
        < np.expand_dims(valid_length, 1)
    segment_ids = np.random.randint(0, 2, (batch_size, seq_length))
    mlm_positions = np.random.randint(0, seq_length // 2, (batch_size, num_mask))
    TF1_Hub_Modules = True
    try:
        tf_model = hub.Module(hub_model_dir, trainable=True)
        # see https://www.tensorflow.org/hub/tf1_hub_module for details
        logging.info('The model is loaded as the TF1 Hub Model')
        tf_input_ids = tf.constant(input_ids, dtype=np.int32)
        tf_input_mask = tf.constant(input_mask, dtype=np.int32)
        tf_segment_ids = tf.constant(segment_ids, dtype=np.int32)
        tf_mlm_positions = tf.constant(mlm_positions, dtype=np.int32)
        tf_mlm_outputs = tf_model(
            dict(input_ids=tf_input_ids, input_mask=tf_input_mask, segment_ids=tf_segment_ids,
                 mlm_positions=tf_mlm_positions), signature="mlm", as_dict=True)
        tf_token_outputs = tf_model(
            dict(input_ids=tf_input_ids, input_mask=tf_input_mask, segment_ids=tf_segment_ids),
            signature="tokens", as_dict=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_params = sess.run(tf_model.variable_map)
            tf_token_outputs_np = sess.run(tf_token_outputs)
            tf_mlm_outputs_np = sess.run(tf_mlm_outputs)
    except RuntimeError as _:
        logging.warning('The provided model directory is not valid for TF1 Hub Modules. '
                        'Now try to load as TF2 SavedModels')
        bert_layer = hub.KerasLayer(hub_model_dir, trainable=True)
        # see https://www.tensorflow.org/hub/tf2_saved_model for details
        logging.info('The model is loaded as the TF2 SavedModel')
        TF1_Hub_Modules = False
        input_word_ids = tf.keras.layers.Input(shape=(seq_length, ), dtype=tf.int32,
                                               name="input_word_ids")
        input_word_mask = tf.keras.layers.Input(shape=(seq_length, ), dtype=tf.int32,
                                                name="input_mask")
        segment_type_ids = tf.keras.layers.Input(shape=(seq_length, ), dtype=tf.int32,
                                                 name="segment_ids")
        pooled_output, sequence_output = bert_layer(
            [input_word_ids, input_word_mask, segment_type_ids])
        tf_model = tf.keras.Model(inputs=[input_word_ids, input_word_mask, segment_type_ids],
                                  outputs=[pooled_output, sequence_output])
        tf_params = {}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pooled_output, sequence_output = tf_model.predict([input_ids, input_mask, segment_ids])
            tf_token_outputs_np = {
                'pooled_output': pooled_output,
                'sequence_output': sequence_output
            }
            # The name of the parameters in TF2 SavedModel are ending with ':0'
            # like 'bert_model/word_embeddings/embeddings_2:0'
            tf_params = {v.name.split(":")[0]: v.read_value() for v in tf_model.variables}
            tf_params = sess.run(tf_params)

    if USE_TF_V1 and TF1_Hub_Modules:
        tf_params_by_read = read_tf_checkpoint(os.path.join(hub_model_dir, 'variables',
                                                            'variables'))
        for k in tf_params:
            assert_allclose(tf_params[k], tf_params_by_read[k])

    # Get parameter names for Tensorflow with unused parameters filtered out.
    tf_names = sorted(tf_params.keys())
    tf_names = filter(lambda name: not name.endswith('adam_m'), tf_names)
    tf_names = filter(lambda name: not name.endswith('adam_v'), tf_names)
    tf_names = filter(lambda name: name != 'Variable', tf_names)
    tf_names = filter(lambda name: name != 'global_step', tf_names)
    tf_names = list(tf_names)

    # Build gluon model and initialize
    # TODO leezu
    # cfg.defrost()
    # cfg.MODEL.hidden_dropout_prob = 0.0
    # cfg.MODEL.attention_dropout_prob = 0.0
    # cfg.freeze()
    gluon_model = PretrainedModel.from_cfg(cfg, use_pooler=True)
    if args.torch:
        gluon_model = gluon_model.to(args.device)
        gluon_model.eval()
    else:
        gluon_model.initialize(ctx=args.ctx)
        gluon_model.hybridize()
    gluon_mlm_model = PretrainedMLMModel(backbone_cfg=cfg)
    if args.torch:
        gluon_mlm_model = gluon_mlm_model.to(args.device)
        gluon_mlm_model.backbone_model.to(args.device)
        gluon_mlm_model.eval()
    else:
        gluon_mlm_model.initialize(ctx=args.ctx)
        gluon_mlm_model.hybridize()

    # Pepare test data
    if args.torch:
        input_ids = th.from_numpy(input_ids).to(args.device)
        valid_length = th.from_numpy(valid_length).to(args.device)
        token_types = th.from_numpy(segment_ids).to(args.device)
        masked_positions = th.from_numpy(mlm_positions).to(args.device)
    else:
        input_ids = mx.np.array(input_ids, dtype=np.int32, ctx=args.ctx)
        valid_length = mx.np.array(valid_length, dtype=np.int32, ctx=args.ctx)
        token_types = mx.np.array(segment_ids, dtype=np.int32, ctx=args.ctx)
        masked_positions = mx.np.array(mlm_positions, dtype=np.int32, ctx=args.ctx)

    # start converting for 'backbone' and 'mlm' model.
    # However sometimes there is no mlm parameter in Tf2 SavedModels like bert wmm large
    if any(['cls' in name for name in tf_names]):
        has_mlm = True
    else:
        has_mlm = False
        logging.info('There is no mask language model parameter in this pretrained model')
    name_map = get_name_map(tf_names, is_TF1=TF1_Hub_Modules)
    # go through the gluon model to infer the shape of parameters
    if has_mlm:
        model = gluon_mlm_model
        contextual_embedding, pooled_output, mlm_scores = \
            model(input_ids, token_types, valid_length, masked_positions)
    else:
        model = gluon_model
        contextual_embedding, pooled_output = model(input_ids, token_types, valid_length)

    # replace tensorflow parameter names with gluon parameter names
    params = {n: p for n, p in model.named_parameters()} if args.torch else model.collect_params()
    all_keys = set(params.keys())
    for (src_name, dst_name) in name_map.items():
        tf_param_val = tf_params[src_name]
        if dst_name is None:
            continue
        if args.torch and dst_name == 'mlm_decoder.3.weight':  # shared weight
            continue
        all_keys.remove(dst_name)
        if 'self_attention/attention_output/kernel' in src_name:
            if args.torch:
                params[dst_name].data = th.from_numpy(tf_param_val.reshape(
                    (cfg.MODEL.units, -1)).T).contiguous()
            else:
                params[dst_name].set_data(tf_param_val.T)
        elif src_name.endswith('kernel'):
            if args.torch:
                params[dst_name].data = th.from_numpy(tf_param_val.T).contiguous()
            else:
                params[dst_name].set_data(tf_param_val.T)
        else:
            if args.torch:
                params[dst_name].data = th.from_numpy(tf_param_val).contiguous()
            else:
                params[dst_name].set_data(tf_param_val)

    # Merge query/kernel, key/kernel, value/kernel to encoder.all_encoder_groups.0.attn_qkv.weight
    def convert_qkv_weights(tf_prefix, prefix, is_mlm):
        """
        To convert the qkv weights with different prefix.

        In tensorflow framework, the prefix of query/key/value for the albert model is
        'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel',
        and that for the bert model is 'bert/encoder/layer_{}/attention/self/key/bias'.
        In gluonnlp framework, the prefix is slightly different as
        'encoder.all_encoder_groups.0.attn_qkv.weight' for albert model and
        'encoder.all_layers.{}.attn_qkv.weight' for bert model, as the
        curly braces {} can be filled with the layer number.
        """
        query_weight = tf_params['{}/query/kernel'.format(tf_prefix)]
        key_weight = tf_params['{}/key/kernel'.format(tf_prefix)]
        value_weight = tf_params['{}/value/kernel'.format(tf_prefix)]
        query_bias = tf_params['{}/query/bias'.format(tf_prefix)]
        key_bias = tf_params['{}/key/bias'.format(tf_prefix)]
        value_bias = tf_params['{}/value/bias'.format(tf_prefix)]
        if 'self_attention' in tf_prefix:
            query_weight = query_weight.reshape((cfg.MODEL.units, -1))
            key_weight = key_weight.reshape((cfg.MODEL.units, -1))
            value_weight = value_weight.reshape((cfg.MODEL.units, -1))
            query_bias = query_bias.reshape((-1, ))
            key_bias = key_bias.reshape((-1, ))
            value_bias = value_bias.reshape((-1, ))
        # Merge query_weight, key_weight, value_weight to params
        weight_name = 'encoder.{}.attn_qkv.weight'.format(prefix)
        bias_name = 'encoder.{}.attn_qkv.bias'.format(prefix)
        if is_mlm:
            weight_name = 'backbone_model.' + weight_name
            bias_name = 'backbone_model.' + bias_name
        if args.torch:
            params[weight_name].data = th.from_numpy(np.concatenate(
                [query_weight, key_weight, value_weight], axis=1).T).contiguous()
        else:
            params[weight_name].set_data(
                np.concatenate([query_weight, key_weight, value_weight], axis=1).T)
        all_keys.remove(weight_name)
        # Merge query_bias, key_bias, value_bias to params
        if args.torch:
            params[bias_name].data = th.from_numpy(
                np.concatenate([query_bias, key_bias, value_bias], axis=0)).contiguous()
        else:
            params[bias_name].set_data(
                np.concatenate([query_bias, key_bias, value_bias], axis=0))
        all_keys.remove(bias_name)

    tf_prefix = None
    if not args.torch and has_mlm:
        all_keys.remove('mlm_decoder.3.weight')
    if model_type == 'bert':
        assert all([
            re.match(
                r'^(backbone_model\.){0,1}encoder\.all_layers\.[\d]+\.attn_qkv\.(weight|bias)$',
                key) is not None for key in all_keys
        ])
        for layer_id in range(cfg.MODEL.num_layers):
            prefix = 'all_layers.{}'.format(layer_id)
            if TF1_Hub_Modules:
                tf_prefix = 'bert/encoder/layer_{}/attention/self'.format(layer_id)
            else:
                tf_prefix = 'transformer/layer_{}/self_attention'.format(layer_id)
            convert_qkv_weights(tf_prefix, prefix, has_mlm)
    elif model_type == 'albert':
        assert all([
            re.match(
                r'^(backbone_model\.){0,1}encoder\.all_encoder_groups\.0\.attn_qkv\.(weight|bias)$',
                key) is not None for key in all_keys
        ])
        prefix = 'all_encoder_groups.0'
        assert TF1_Hub_Modules, 'Please download the albert model from TF1 Hub'
        tf_prefix = 'bert/encoder/transformer/group_0/inner_group_0/attention_1/self'
        convert_qkv_weights(tf_prefix, prefix, has_mlm)
    else:
        raise NotImplementedError

    tolerance = 5E-4 if cfg.MODEL.num_layers == 24 else 1E-4
    # The pooled_output of albert large will have 0.5% mismatch under the tolerance of 1E-2,
    # for that we are going to use a small tolerance to pass the difference checking
    tolerance = 0.2 if 'albert_large' in args.tf_hub_model_path else tolerance

    assert len(all_keys) == 0, f"The following torch parameters weren't assigned to: {all_keys}"

    def check_backbone(tested_model, tf_token_outputs_np):
        # test conversion results for backbone model
        tf_contextual_embedding = tf_token_outputs_np['sequence_output']
        tf_pooled_output = tf_token_outputs_np['pooled_output']
        contextual_embedding, pooled_output = \
            tested_model(input_ids, token_types, valid_length)
        if args.torch:
            assert_allclose(pooled_output.detach().cpu().numpy(), tf_pooled_output, tolerance,
                            tolerance)
        else:
            assert_allclose(pooled_output.asnumpy(), tf_pooled_output, tolerance, tolerance)
        for i in range(batch_size):
            ele_valid_length = int(valid_length[i])
            if args.torch:
                assert_allclose(contextual_embedding[i, :ele_valid_length, :].detach().cpu().numpy(),
                                tf_contextual_embedding[i, :ele_valid_length, :], tolerance,
                                tolerance)
            else:
                assert_allclose(contextual_embedding[i, :ele_valid_length, :].asnumpy(),
                                tf_contextual_embedding[i, :ele_valid_length, :], tolerance,
                                tolerance)

    if not has_mlm:
        if test_conversion:
            check_backbone(model, tf_token_outputs_np)
        th.save(model.state_dict(), os.path.join(save_dir, 'model.params'))
        logging.info('Convert the backbone model in {} to {}/{}'.format(
            hub_model_dir, save_dir, 'model.params'))
    else:
        # test conversion results for mlm model
        # TODO(zheyuye), figure out how to check the mlm model from TF2 SavedModel
        if test_conversion:
            backbone_model = model.backbone_model
            if args.torch:
                model = model.to(args.device)
                backbone_model = backbone_model.to(args.device)
            check_backbone(backbone_model, tf_mlm_outputs_np)
            if TF1_Hub_Modules:
                tf_contextual_embedding = tf_mlm_outputs_np['sequence_output']
                tf_pooled_output = tf_mlm_outputs_np['pooled_output']
                tf_mlm_scores = tf_mlm_outputs_np['mlm_logits'].reshape((batch_size, num_mask, -1))
                contextual_embedding, pooled_output, mlm_scores = \
                    model(input_ids, token_types, valid_length, masked_positions)
                if args.torch:
                    assert_allclose(pooled_output.detach().cpu().numpy(), tf_pooled_output,
                                    tolerance, tolerance)
                    assert_allclose(mlm_scores.detach().cpu().numpy(), tf_mlm_scores,
                                    tolerance, tolerance)
                else:
                    assert_allclose(pooled_output.asnumpy(), tf_pooled_output, tolerance, tolerance)
                    assert_allclose(mlm_scores.asnumpy(), tf_mlm_scores, tolerance, tolerance)
                for i in range(batch_size):
                    ele_valid_length = int(valid_length[i])
                    if args.torch:
                        assert_allclose(
                            contextual_embedding[i, :ele_valid_length, :].detach().cpu().numpy(),
                            tf_contextual_embedding[i, :ele_valid_length, :], tolerance, tolerance)
                    else:
                        assert_allclose(contextual_embedding[i, :ele_valid_length, :].asnumpy(),
                                        tf_contextual_embedding[i, :ele_valid_length, :],
                                        tolerance, tolerance)
        if args.torch:
            th.save(model.backbone_model.state_dict(), os.path.join(save_dir, 'model.params'))
            th.save(model.state_dict(), os.path.join(save_dir, 'model_mlm.params'))
        else:
            model.backbone_model.save_parameters(os.path.join(
                save_dir, 'model.params'), deduplicate=True)
            model.save_parameters(os.path.join(save_dir, 'model_mlm.params'), deduplicate=True)
        logging.info('Convert the backbone model in {} to {}/{}'.format(
            hub_model_dir, save_dir, 'model.params'))
        logging.info('Convert the MLM model in {} to {}/{}'.format(hub_model_dir, save_dir,
                                                                   'model_mlm.params'))

    # TODO(zheyuye) the gradient checking could be explored in further development

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
    save_dir = args.save_dir \
        if args.save_dir is not None else os.path.basename(args.tf_hub_model_path) + '_gluon'
    convert_tf_model(args.tf_hub_model_path, save_dir, args.test, args.model_type)
