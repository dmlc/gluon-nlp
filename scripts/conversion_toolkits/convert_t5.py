import argparse
import logging
import json
import os
import shutil
import tempfile

import torch
import mxnet as mx
from mxnet import np, npx
from gluonnlp.attention_cell import gen_self_attn_mask, gen_mem_attn_mask
from gluonnlp.data.tokenizers import SentencepieceTokenizer
from gluonnlp.models.t5 import T5Model as Gluon_T5
from gluonnlp.utils.misc import download, logging_config, sha1sum, naming_convention
from transformers import T5Model as HF_T5


# these mappings are adapted from huggingface T5 folder
T5_PRETRAINED_MODEL_MAP = {
    "t5-small": "google_t5_small",
    "t5-base": "google_t5_base",
    "t5-large": "google_t5_large",
    "t5-3b": "google_t5_3B",
    "t5-11b": "google_t5_11B"
}
T5_PRETRAINED_CONFIG_MAP = {
    "t5-small": "https://huggingface.co/t5-small/resolve/main/config.json",
    "t5-base": "https://huggingface.co/t5-base/resolve/main/config.json",
    "t5-large": "https://huggingface.co/t5-large/resolve/main/config.json",
    "t5-3b": "https://huggingface.co/t5-3b/resolve/main/config.json",
    "t5-11b": "https://huggingface.co/t5-11b/resolve/main/config.json"
}
PRETRAINED_VOCAB_MAP = {
    "t5-small": "https://huggingface.co/t5-small/resolve/main/spiece.model",
    "t5-base": "https://huggingface.co/t5-base/resolve/main/spiece.model",
    "t5-large": "https://huggingface.co/t5-large/resolve/main/spiece.model",
    "t5-3b": "https://huggingface.co/t5-3b/resolve/main/spiece.model",
    "t5-11b": "https://huggingface.co/t5-11b/resolve/main/spiece.model"
}


# this mapping only works on "T5Model" class from Huggingface and GluonNLP
PARAM_MAP = [
    # 0.
    ('shared.weight', 'input_embedding_layer.weight'), 
    # 1. encoder / decoder
    ('{}.block.0.layer.0.SelfAttention.relative_attention_bias.weight', '{}.relative_position_encoder._rel_pos_embed.weight'), 
    # 2. encoder / decoder, block/layer #, 0->self_attn_layer_norm / (decoder: 1->cross_attn_layer_norm) / (encoder: 1/decoder: 2)->ffn.layer_norm
    ('{}.block.{}.layer.{}.layer_norm.weight', '{}.layers.{}.{}.gamma'), 
    # 3. encoder / decoder, block/layer #, 0.Self->self / 1.EncDec->cross, q/k/v
    ('{}.block.{}.layer.{}Attention.{}.weight', '{}.layers.{}.{}_attn_{}.weight'), 
    # 4. encoder / decoder, block/layer #, 0.SelfAttention.o->self_attn_proj / (decoder: 1.EncDecAttention.o->cross_attn_proj)
    ('{}.block.{}.layer.{}.weight', '{}.layers.{}.{}.weight'), 
    # 5. encoder / decoder, block/layer #, (encoder: 1 / decoder: 2), wi->ffn_1 / wi_0->gated_ffn_1 / wi_1->ffn_1 / wo->ffn_2
    ('{}.block.{}.layer.{}.DenseReluDense.{}.weight', '{}.layers.{}.ffn.{}.weight'), 
    # 6. encoder / decoder
    ('{}.final_layer_norm.weight', '{}.final_layer_norm.gamma'), 
]


def parse_args(): 
    parser = argparse.ArgumentParser('Convert Huggingface T5 Model to GluonNLP')
    parser.add_argument(
        'model_name', choices=list(T5_PRETRAINED_MODEL_MAP.keys()), help='Name of pretrained T5 model in Huggingface.'
    )
    parser.add_argument(
        'dest_dir', help='Directory to save converted config, vocab and weights.'
    )
    parser.add_argument(
        '--test', action='store_true', required=False, default=False, help='Whether to test conversion correctness.'
    )
    return parser.parse_args()


def convert_config(args, converted): 
    print('converting cfg...')
    # download config
    gluon_cfg = Gluon_T5.get_cfg(T5_PRETRAINED_MODEL_MAP[args.model_name])
    with tempfile.TemporaryDirectory() as temp_dir: 
        hf_cfg_path = os.path.join(temp_dir, 'config.json')
        download(
            url=T5_PRETRAINED_CONFIG_MAP[args.model_name], 
            path=hf_cfg_path
        )
        with open(hf_cfg_path, 'r') as f: 
            hf_cfg = json.load(f)
        os.remove(hf_cfg_path)
    # update attributes
    cfg = gluon_cfg.clone()
    cfg.defrost()
    cfg.MODEL.vocab_size = hf_cfg['vocab_size']
    cfg.MODEL.d_model = hf_cfg['d_model']
    cfg.MODEL.d_kv = hf_cfg['d_kv']
    cfg.MODEL.d_ff = hf_cfg['d_ff']
    cfg.MODEL.num_layers = hf_cfg['num_layers']
    cfg.MODEL.num_heads = hf_cfg['num_heads']
    cfg.MODEL.layer_norm_eps = hf_cfg['layer_norm_epsilon']
    cfg.MODEL.dropout_prob = hf_cfg['dropout_rate']
    cfg.INITIALIZER.init_factor = hf_cfg['initializer_factor']
    cfg.freeze()
    # save config
    config_path = os.path.join(args.dest_dir, 'model.yml')
    with open(config_path, 'w') as f: 
        f.write(cfg.dump())
    converted['config'] = config_path
    return cfg


def convert_vocab(args, converted): 
    print('converting vocab...')
    # at this step we don't add <extra_id>s into the vocab, but just save the original binary file directly
    # those special tokens are added only when instantiating a T5Tokenizer
    vocab_path = os.path.join(args.dest_dir, 't5.vocab')
    download(
        url=PRETRAINED_VOCAB_MAP[args.model_name], 
        path=vocab_path
    )
    converted['vocab'] = vocab_path


def convert_params(args, converted, hf_model, gluon_model): 
    print('converting parameters...')
    # prepare models and parameters
    gluon_model.initialize()
    hf_params = hf_model.state_dict()
    gluon_params = gluon_model.collect_params()
    # TODO(yongyi-wu): add sanity check, eg. param #, layer #, ffn activation, etc.
    num_layers = gluon_model.num_layers

    def convert(hf_param, gluon_param): 
        gluon_params[gluon_param].set_data(hf_params[hf_param].cpu().numpy())
    
    # convert parameters
    for idx, (hf_key, gluon_key) in enumerate(PARAM_MAP): 
        if idx == 0: 
            convert(hf_key, gluon_key)    
        elif idx == 1: 
            for i in ['encoder', 'decoder']: 
                convert(hf_key.format(i), gluon_key.format(i))
        elif idx in [2, 3, 4, 5]: 
            for stack in ['encoder', 'decoder']: 
                for layer in range(num_layers): 
                    if idx == 2: 
                        if stack == 'encoder': 
                            L = ['self_attn_layer_norm', 'ffn.layer_norm']
                        else: 
                            L = ['self_attn_layer_norm', 'cross_attn_layer_norm', 'ffn.layer_norm']
                        for i, j in enumerate(L): 
                            convert(hf_key.format(stack, layer, i), gluon_key.format(stack, layer, j))
                    elif idx == 3: 
                        for i in ['q', 'k', 'v']: 
                            convert(
                                hf_key.format(stack, layer, '0.Self', i), 
                                gluon_key.format(stack, layer, 'self', i)
                            )
                            if stack == 'decoder': 
                                convert(
                                    hf_key.format(stack, layer, '1.EncDec', i), 
                                    gluon_key.format(stack, layer, 'cross', i)
                                )
                    elif idx == 4: 
                        convert(
                            hf_key.format(stack, layer, '0.SelfAttention.o'), 
                            gluon_key.format(stack, layer, 'self_attn_proj')
                        )
                        if stack == 'decoder': 
                            convert(
                                hf_key.format(stack, layer, '1.EncDecAttention.o'), 
                                gluon_key.format(stack, layer, 'cross_attn_proj')
                            )
                    elif idx == 5:
                        if gluon_model.activation == 'relu': 
                            denses = [('wi', 'ffn_1'), ('wo', 'ffn_2')]
                        elif gluon_model.activation == 'gated-gelu': 
                            denses = [('wi_0', 'gated_ffn_1'), ('wi_1', 'ffn_1'), ('wo', 'ffn_2')]
                        else: 
                            raise ValueError
                        i = 1 if stack == 'encoder' else 2
                        for j1, j2 in denses: 
                            convert(
                                hf_key.format(stack, layer, i, j1), 
                                gluon_key.format(stack, layer, j2)
                        )
        elif idx == 6: 
            for stack in ['encoder', 'decoder']: 
                convert(hf_key.format(stack), gluon_key.format(stack))
    # save parameters
    param_path = os.path.join(args.dest_dir, 'model.params')
    gluon_model.save_parameters(param_path)
    converted['params'] = param_path
    return gluon_model


def rename(args, converted): 
    for item, old_path in converted.items(): 
        new_name, _ = naming_convention(args.dest_dir, os.path.basename(old_path))
        new_path = os.path.join(args.dest_dir, new_name)
        shutil.move(old_path, new_path)
        logging.info('{} of {} has been converted to {}.'.format(item, args.model_name, new_path))
        converted[item] = new_path


def test_conversion(args, hf_model, gluon_model): 
    logging.info('testing conversion...')
    # create dummy input
    batch_size = 6
    src_length = 128
    tgt_length = 8
    vocab_size = hf_model.shared.weight.shape[0]
    src_data = np.random.randint(1, vocab_size, (batch_size, src_length))
    src_valid_length = np.random.randint(src_length // 2, src_length, (batch_size,))
    tgt_data = np.random.randint(1, vocab_size, (batch_size, tgt_length))
    tgt_valid_length = np.random.randint(tgt_length // 2, tgt_length, (batch_size,))
    enc_attn_mask = npx.arange_like(src_data, axis=-1) < src_valid_length.reshape(-1, 1)
    dec_attn_mask = npx.arange_like(tgt_data, axis=-1) < tgt_valid_length.reshape(-1, 1)
    # test T5Model forward pass
    hf_model.eval() # disable dropout
    hf_out = hf_model(
        input_ids=torch.from_numpy(src_data.asnumpy()),  
        attention_mask=torch.from_numpy(enc_attn_mask.asnumpy()), 
        decoder_input_ids=torch.from_numpy(tgt_data.asnumpy()), 
        decoder_attention_mask=torch.from_numpy(dec_attn_mask.asnumpy())
    )['last_hidden_state'].detach().numpy()
    gl_out = gluon_model(src_data, src_valid_length, tgt_data, tgt_valid_length)
    for i in  range(batch_size):
        assert np.allclose(
            hf_out[i, :tgt_valid_length[i].item(), :], 
            gl_out[i, :tgt_valid_length[i].item(), :], 
            1E-3, 
            1E-3
        )
    logging.info('pass')


def convert_t5(args): 
    logging.info('converting T5 model from Huggingface...')
    if not os.path.exists(args.dest_dir): 
        os.mkdir(args.dest_dir)
    converted = {}
    # convert and save vocab
    convert_vocab(args, converted)
    # convert and save config
    gluon_cfg = convert_config(args, converted)
    # convert, (test), and save model
    hf_t5 = HF_T5.from_pretrained(args.model_name)
    gluon_t5 = Gluon_T5.from_cfg(gluon_cfg)
    gluon_t5 = convert_params(args, converted, hf_t5, gluon_t5)
    gluon_t5.hybridize()
    # test model if needed
    if args.test: 
        test_conversion(args, hf_t5, gluon_t5)
    # rename with sha1sum
    rename(args, converted)
    logging.info('conversion completed.')
    logging.info('file statistics:')
    for item, new_path in converted.items(): 
        logging.info('filename: {}\tsize: {}\tsha1sum: {}'.format(
            os.path.basename(new_path), os.path.getsize(new_path), sha1sum(new_path)
        ))
    return converted


if __name__ == "__main__": 
    args = parse_args()
    logging_config()
    convert_t5(args)
