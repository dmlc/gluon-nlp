import argparse
import json
import os
import re
import shutil
import collections
import io
import logging
from numpy.testing import assert_allclose
import mxnet as mx
import numpy as np
from gluonnlp.utils.misc import sha1sum, logging_config
from gluonnlp.models.bert import BertModel, BertForMLM
from gluonnlp.models.albert import AlbertModel, AlbertForMLM
from gluonnlp.data.tokenizers import SentencepieceTokenizer
from gluonnlp.data.vocab import Vocab
import tensorflow
USE_TF_V1 = tensorflow.version.VERSION.split('.')[0] < '2'
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

mx.npx.set_np()
np.random.seed(1234)
mx.npx.random.seed(1234)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the TF pretrained model to Gluon')
    parser.add_argument('--tf_hub_model_path', type=str,
                        help='Directory of the model downloaded from TF hub.')
    parser.add_argument('--model_type', type=str, choices=['bert','albert'],
                        help='The name of the model to be converted. Only Bert and Albert are currently supported.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='directory path to save the converted pretrained model.')
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


def convert_tf_config(json_cfg_path, vocab_size, model_type):
    """Convert the config file"""

    with open(json_cfg_path, encoding='utf-8') as f:
        json_cfg = json.load(f)
    if model_type == 'bert':
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
    cfg.MODEL.dtype = 'float32'
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
        if ele.endswith('.txt'):
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
        vocab_size = len(open(vocab_path, 'rU').readlines())
    cfg = convert_tf_config(json_cfg_path, vocab_size, model_type)
    return cfg, vocab_path, spm_model_path

CONVERT_MAP_TF1 = [
    ('bert/', ''),
    ('cls/', ''),
    ('predictions/output_bias', 'word_embed_bias'),
    ('predictions', 'mlm'),
    ('transform/dense', 'proj'),
    ('transformer/', ''),
    ('transform/', ''),
    ('embeddings/word_embeddings', 'word_embed_weight'),
    ('embeddings/token_type_embeddings', 'token_type_embed_weight'),
    ('embeddings/position_embeddings', 'token_pos_embed_embed_weight'),
    ('encoder/embedding_hidden_mapping_in', 'embed_factorized_proj'),
    ('encoder', 'enc'),
    ('inner_group_0/', ''),
    ('group', 'groups'),
    ('layer', 'layers'),
    ('embeddings', 'embed'),
    ('attention/output/LayerNorm', 'ln'), #bert
    ('output/LayerNorm', 'ffn_ln'), #bert
    ('LayerNorm_1', 'ffn_ln'), #albert
    ('LayerNorm', 'ln'),  #albert
    ('ffn_1/', ''),
    ('attention_1', 'attention'), #albert
    ('attention/output/dense', 'proj'),
    ('intermediate/dense', 'ffn_ffn1'),
    ('intermediate/output/dense', 'ffn_ffn2'), #albert
    ('output/dense', 'ffn_ffn2'), #bert
    ('output/', ''),
    ('pooler/dense', 'pooler'),
    ('kernel', 'weight'),
    ('attention/', ''),
    ('/', '_'),
]

CONVERT_MAP_TF2 = [
    (':0', ''),
    ('cls/', ''),
    ('bert_model/', ''),
    ('predictions/output_bias', 'word_embed_bias'),
    ('predictions', 'mlm'),
    ('word_embeddings/embeddings', 'word_embed_weight'),
    ('embedding_postprocessor/type_embeddings', 'token_type_embed_weight'), #bert
    ('embedding_postprocessor/position_embeddings', 'token_pos_embed_embed_weight'), #bert
    ('embedding_postprocessor/layer_norm', 'embed_ln'), #bert
    ('position_embedding/embeddings', 'token_pos_embed_embed_weight'), #albert
    ('type_embeddings/embeddings', 'token_type_embed_weight'), #albert
    ('embeddings/layer_norm', 'embed_ln'), #albert
    ('embedding_projection', 'embed_factorized_proj'),
    ('transformer', 'enc_groups_0'),
    ('self_attention_output', 'proj'),
    ('self_attention_layer_norm', 'ln'),
    ('intermediate', 'ffn_ffn1'),
    ('output_layer_norm', 'ffn_ln'),
    ('output', 'ffn_ffn2'),
    ("pooler_transform", "pooler"),
    ('encoder', 'enc'),
    ('layer', 'layers'),
    ('kernel', 'weight'),
    ('/', '_'),
]


def get_name_map(tf_names, is_mlm=False, is_TF1=True):
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
    is_mlm
        wether a mask language model
    is_TF1
        whether load from TF1 Hub Modules
    Returns
    -------
    A dictionary with the following format:
        {tf_names : mx_names}
    """
    convert_map = CONVERT_MAP_TF1 if is_TF1 else CONVERT_MAP_TF2
    name_map = {}
    for source_name in tf_names:
        target_name = source_name
        if not is_mlm and 'cls' in source_name:
            continue
        # skip the qkv weights
        if 'self/' in source_name:
            name_map[source_name] = None
            continue
        if 'self_attention/' in source_name:
            name_map[source_name] = None
            continue
        for old, new in convert_map:
            target_name = target_name.replace(old, new)
        name_map[source_name] = target_name
    return name_map


def convert_tf_model(hub_model_dir, save_dir, test_conversion, model_type, gpu):
    ctx = mx.gpu(gpu) if gpu is not None else mx.cpu()
    # set up the model type to be converted
    if model_type == 'bert':
        PretrainedModel, PretrainedMLMModel = BertModel, BertForMLM
    elif model_type == 'albert':
        PretrainedModel, PretrainedMLMModel = AlbertModel, AlbertForMLM
    else:
        raise NotImplementedError

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # a temporary folder to save converted files
    tmp_dir = os.path.expanduser(os.path.join(save_dir, 'tmp'))
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    cfg, vocab_path, spm_model_path = convert_tf_assets(os.path.join(hub_model_dir, 'assets'),
                                                        model_type)
    with open(os.path.join(tmp_dir, 'model.yml'), 'w') as of:
        of.write(cfg.dump())
    if spm_model_path:
        tokenizer = SentencepieceTokenizer(spm_model_path)
        new_vocab = Vocab(tokenizer.vocab.all_tokens,
                          unk_token='<unk>',
                          pad_token='<pad>',
                          cls_token='[CLS]',
                          sep_token='[SEP]',
                          mask_token='[MASK]')
        new_vocab.save(os.path.join(tmp_dir, 'vocab.json'))
        shutil.copy(spm_model_path, os.path.join(tmp_dir, 'spm.model'))
    elif vocab_path:
        shutil.copy(vocab_path, os.path.join(tmp_dir, 'vocab.json'))

    #test input data
    batch_size = 2
    seq_length = 16
    num_mask = 5
    input_ids = np.random.randint(0, cfg.MODEL.vocab_size, (batch_size, seq_length))
    valid_length = np.random.randint(seq_length // 2, seq_length, (batch_size,))
    input_mask = np.broadcast_to(np.arange(seq_length).reshape(1, -1), (batch_size, seq_length))\
                 < np.expand_dims(valid_length, 1)
    segment_ids = np.random.randint(0, 2, (batch_size, seq_length))
    mlm_positions = np.random.randint(0, seq_length // 2, (batch_size, num_mask))
    TF1_Hub_Modules = True
    try:
        tf_model = hub.Module(hub_model_dir, trainable=True)
        #see https://www.tensorflow.org/hub/tf1_hub_module for details
        logging.info('The model is loaded as the TF1 Hub Model')
        tf_input_ids = tf.constant(input_ids, dtype=np.int32)
        tf_input_mask = tf.constant(input_mask, dtype=np.int32)
        tf_segment_ids = tf.constant(segment_ids, dtype=np.int32)
        tf_mlm_positions = tf.constant(mlm_positions, dtype=np.int32)
        tf_mlm_outputs = tf_model(
            dict(input_ids=tf_input_ids,
                 input_mask=tf_input_mask,
                 segment_ids=tf_segment_ids,
                 mlm_positions=tf_mlm_positions), signature="mlm", as_dict=True)
        tf_token_outputs = tf_model(
            dict(input_ids=tf_input_ids,
                 input_mask=tf_input_mask,
                 segment_ids=tf_segment_ids), signature="tokens", as_dict=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_params = sess.run(tf_model.variable_map)
            tf_token_outputs_np = sess.run(tf_token_outputs)
            tf_mlm_outputs_np = sess.run(tf_mlm_outputs)
    except RuntimeError as _:
        logging.warning('The provided model directory is not valid for TF1 Hub Modules. '
                        'Now try to load as TF2 SavedModels')
        bert_layer = hub.KerasLayer(hub_model_dir, trainable=True)
        #see https://www.tensorflow.org/hub/tf2_saved_model for details
        logging.info('The model is loaded as the TF2 SavedModel')
        TF1_Hub_Modules = False
        input_word_ids = tf.keras.layers.Input(shape=(seq_length), dtype=tf.int32,
                                               name="input_word_ids")
        input_word_mask = tf.keras.layers.Input(shape=(seq_length), dtype=tf.int32,
                                                name="input_mask")
        segment_type_ids = tf.keras.layers.Input(shape=(seq_length), dtype=tf.int32,
                                                 name="segment_ids")
        pooled_output, sequence_output = bert_layer([input_word_ids, input_word_mask,
                                                     segment_type_ids])
        tf_model = tf.keras.Model(
                inputs=[input_word_ids, input_word_mask, segment_type_ids],
                outputs=[pooled_output, sequence_output]
                )
        tf_params = {}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pooled_output, sequence_output = tf_model.predict([input_ids, input_mask, segment_ids])
            tf_token_outputs_np = {'pooled_output': pooled_output,
                                   'sequence_output': sequence_output}
            # The name of the parameters in TF2 SavedModel are ending with ':0'
            # like 'bert_model/word_embeddings/embeddings_2:0'
            tf_params = {v.name.split(":")[0]: v.read_value() for v in tf_model.variables}
            tf_params = sess.run(tf_params)

    if USE_TF_V1 and TF1_Hub_Modules:
        tf_params_by_read = read_tf_checkpoint(os.path.join(hub_model_dir, 'variables', 'variables'))
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
    gluon_model = PretrainedModel.from_cfg(cfg, prefix='', use_pooler=True)
    gluon_model.initialize(ctx=ctx)
    gluon_model.hybridize()
    gluon_mlm_model = PretrainedMLMModel(backbone_cfg=cfg, prefix='')
    gluon_mlm_model.initialize(ctx=ctx)
    gluon_mlm_model.hybridize()

    # Pepare test data
    mx_input_ids = mx.np.array(input_ids, dtype=np.int32, ctx=ctx)
    mx_valid_length = mx.np.array(valid_length, dtype=np.int32, ctx=ctx)
    mx_token_types = mx.np.array(segment_ids, dtype=np.int32, ctx=ctx)
    mx_masked_positions = mx.np.array(mlm_positions, dtype=np.int32, ctx=ctx)

    # start converting for 'backbone' and 'mlm' model.
    # However sometimes there is no mlm parameter in Tf2 SavedModels like bert wmm large
    if any(['cls' in name for name in tf_names]):
        is_mlms = [False, True]
    else:
        is_mlms = [False]
        logging.info('There is no mask language model parameter in this pretrained model')
    for is_mlm in is_mlms:
        name_map = get_name_map(tf_names, is_mlm=is_mlm, is_TF1=TF1_Hub_Modules)
        # go through the gluon model to infer the shape of parameters
        if is_mlm:
            model = gluon_mlm_model
            contextual_embedding, pooled_output, mlm_scores = \
                model(mx_input_ids, mx_token_types, mx_valid_length, mx_masked_positions)
        else:
            model = gluon_model
            contextual_embedding, pooled_output = model(mx_input_ids, mx_token_types,
                                                        mx_valid_length)


        # replace tensorflow parameter names with gluon parameter names
        mx_params = model.collect_params()
        all_keys = set(mx_params.keys())
        for (src_name, dst_name) in name_map.items():
            tf_param_val = tf_params[src_name]
            if dst_name is None:
                continue
            all_keys.remove(dst_name)
            if 'self_attention_output/kernel' in src_name:
                mx_params[dst_name].set_data(tf_param_val.reshape((cfg.MODEL.units, -1)).T)
                continue
            if src_name.endswith('kernel'):
                mx_params[dst_name].set_data(tf_param_val.T)
            else:
                mx_params[dst_name].set_data(tf_param_val)

        # Merge query/kernel, key/kernel, value/kernel to enc_groups_0_attn_qkv_weight
        def convert_qkv_weights(tf_prefix, mx_prefix):
            """
            To convert the qkv weights with different prefix.

            In tensorflow framework, the prefix of query/key/value for the albert model is
            'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel',
            and that for the albert model is 'bert/encoder/layer_{}/attention/self/key/bias'.
            In gluonnlp framework, the prefix is slightly different as 'enc_groups_0_attn_qkv_weight'
            for albert model and 'enc_layers_{}_attn_qkv_weight' for bert model, as the
            curly braces {} can be filled with the layer number.
            """
            query_weight = tf_params[
                '{}/query/kernel'.format(tf_prefix)]
            key_weight = tf_params[
                '{}/key/kernel'.format(tf_prefix)]
            value_weight = tf_params[
                '{}/value/kernel'.format(tf_prefix)]
            query_bias = tf_params[
                '{}/query/bias'.format(tf_prefix)]
            key_bias = tf_params[
                '{}/key/bias'.format(tf_prefix)]
            value_bias = tf_params[
                '{}/value/bias'.format(tf_prefix)]
            if 'self_attention' in tf_prefix:
                query_weight = query_weight.reshape((cfg.MODEL.units, -1))
                key_weight = key_weight.reshape((cfg.MODEL.units, -1))
                value_weight = value_weight.reshape((cfg.MODEL.units, -1))
                query_bias = query_bias.reshape((-1, ))
                key_bias = key_bias.reshape((-1, ))
                value_bias = value_bias.reshape((-1, ))
            # Merge query_weight, key_weight, value_weight to mx_params
            mx_params['enc_{}_attn_qkv_weight'.format(mx_prefix)].set_data(
                np.concatenate([query_weight, key_weight, value_weight], axis=1).T)
            # Merge query_bias, key_bias, value_bias to mx_params
            mx_params['enc_{}_attn_qkv_bias'.format(mx_prefix)].set_data(
                np.concatenate([query_bias, key_bias, value_bias], axis=0))

        tf_prefix = None
        if model_type == 'bert':
            assert all([re.match(r'^enc_layers_[\d]+_attn_qkv_(weight|bias)$',key)
                        is not None for key in all_keys])
            for layer_id in range(cfg.MODEL.num_layers):
                mx_prefix = 'layers_{}'.format(layer_id)
                if TF1_Hub_Modules:
                    tf_prefix = 'bert/encoder/layer_{}/attention/self'.format(layer_id)
                else:
                    tf_prefix = 'bert_model/encoder/layer_{}/self_attention'.format(layer_id)
                convert_qkv_weights(tf_prefix, mx_prefix)
        elif model_type == 'albert':
            assert all_keys == {'enc_groups_0_attn_qkv_weight', 'enc_groups_0_attn_qkv_bias'}
            mx_prefix = 'groups_0'
            if TF1_Hub_Modules:
                tf_prefix = 'bert/encoder/transformer/group_0/inner_group_0/attention_1/self'
            else:
                tf_prefix = 'transformer/self_attention'
            convert_qkv_weights(tf_prefix, mx_prefix)

        else:
            raise NotImplementedError

        if not is_mlm:
            #test conversion results for backbone model
            if test_conversion:
                tf_contextual_embedding = tf_token_outputs_np['sequence_output']
                tf_pooled_output = tf_token_outputs_np['pooled_output']
                contextual_embedding, pooled_output = model(mx_input_ids, mx_token_types, mx_valid_length)
                assert_allclose(pooled_output.asnumpy(), tf_pooled_output, 1E-3, 1E-3)
                for i in range(batch_size):
                    ele_valid_length = valid_length[i]
                    assert_allclose(contextual_embedding[i, :ele_valid_length, :].asnumpy(),
                                    tf_contextual_embedding[i, :ele_valid_length, :], 1E-3, 1E-3)
            model.save_parameters(os.path.join(tmp_dir, 'model.params'), deduplicate=True)
            logging.info('Convert the backbone model in {} to {}/{}'.format(hub_model_dir,
                                                                            tmp_dir, 'model.params'))
        elif is_mlm:
            #test conversion results for mlm model
            #TODO, figure out how to check the mlm model from TF2 SavedModel
            if test_conversion and TF1_Hub_Modules:
                tf_contextual_embedding = tf_mlm_outputs_np['sequence_output']
                tf_pooled_output = tf_mlm_outputs_np['pooled_output']
                tf_mlm_scores = tf_mlm_outputs_np['mlm_logits'].reshape((batch_size, num_mask, -1))
                contextual_embedding, pooled_output, mlm_scores = model(mx_input_ids, mx_token_types, mx_valid_length, mx_masked_positions)
                assert_allclose(pooled_output.asnumpy(), tf_pooled_output, 1E-3, 1E-3)
                assert_allclose(mlm_scores.asnumpy(), tf_mlm_scores, 1E-3, 1E-3)
                for i in range(batch_size):
                    ele_valid_length = valid_length[i]
                    assert_allclose(contextual_embedding[i, :ele_valid_length, :].asnumpy(),
                                    tf_contextual_embedding[i, :ele_valid_length, :], 1E-3, 1E-3)
            model.save_parameters(os.path.join(tmp_dir, 'model_mlm.params'), deduplicate=True)
            logging.info('Convert the MLM model in {} to {}/{}'.format(hub_model_dir,
                                                                       tmp_dir, 'model_mlm.params'))
        else:
            raise NotImplementedError

        #TODO(zheyuye) the gradient checking could be explored in further development

    # naming convention and
    def get_new_name(origin_folder, file_name):
        long_hash = sha1sum(os.path.join(origin_folder, file_name))
        file_prefix, file_sufix = file_name.split('.')
        new_name = '{file_prefix}-{short_hash}.{file_sufix}'.format(
                file_prefix=file_prefix,
                short_hash=long_hash[:8],
                file_sufix=file_sufix)
        return new_name, long_hash

    logging.info('Conversion finished!')
    logging.info('Statistics:')

    file_names = os.listdir(tmp_dir)
    for file_name in file_names:
        new_name, long_hash = get_new_name(tmp_dir, file_name)
        tep_file = os.path.join(tmp_dir, file_name)
        coverted_file = os.path.join(save_dir, new_name)
        shutil.copy(tep_file, coverted_file)
        file_size = os.path.getsize(coverted_file)
        logging.info('\t{}/{} {} {}'.format(save_dir, new_name, long_hash, file_size))


if __name__ == '__main__':
    args = parse_args()
    logging_config()
    save_dir = args.save_dir \
        if args.save_dir is not None else os.path.basename(args.tf_hub_model_path) + '_gluon'
    convert_tf_model(args.tf_hub_model_path, save_dir, args.test, args.model_type, args.gpu)
