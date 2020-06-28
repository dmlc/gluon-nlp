import os
import sys
import argparse
import shutil
import logging
import json
from numpy.testing import assert_allclose
import mxnet as mx
import numpy as np
import torch
from gluonnlp.utils.misc import sha1sum, logging_config
from gluonnlp.models.roberta import RobertaModel as gluon_RobertaModel
from gluonnlp.data.tokenizers import HuggingFaceByteBPETokenizer
from gluonnlp.data.vocab import Vocab as gluon_Vocab
from fairseq.models.roberta import RobertaModel as fairseq_RobertaModel

mx.npx.set_np()

def parse_args():
    parser = argparse.ArgumentParser(description='Convert the fairseq RoBERTa Model to Gluon.')
    parser.add_argument('--fairseq_model_path', type=str, required=True,
                        help='Directory of the fairseq RoBERTa model.')
    parser.add_argument('--model_size', type=str, choices=['base', 'large'], default='base',
                        help='Size of RoBERTa model.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory path to save the converted RoBERTa model.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='The single gpu to run mxnet, (e.g. --gpu 0) the default device is cpu.')
    parser.add_argument('--test', action='store_true',
                        help='Whether to test the conversion.')
    return parser.parse_args()

def convert_vocab(args, fairseq_model):
    print('converting vocab')
    fairseq_dict_path = os.path.join(args.fairseq_model_path, 'dict.txt')
    merges_save_path = os.path.join(args.save_dir, 'gpt2.merges')
    vocab_save_path =os.path.join(args.save_dir, 'gpt2.vocab')
    fairseq_vocab = fairseq_model.task.dictionary
    # bos_word attr missing in fairseq_vocab
    fairseq_vocab.bos_word = fairseq_vocab[fairseq_vocab.bos_index]

    assert os.path.exists(fairseq_dict_path), \
           '{} not found'.format(fairseq_dict_path)
    from mxnet.gluon.utils import download
    temp_vocab_file = download(
        'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json')
    temp_merges_file = download(
        'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe')
    # copy merges directy
    shutil.copy(temp_merges_file, merges_save_path)

    # build vocab
    transfer_dict = []
    with open(fairseq_dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            word_id, count = line.split(' ', 1)
            transfer_dict.append(word_id)
    transfer_dict = {transfer_dict[i] : i for i in range(len(transfer_dict))}
    with open(temp_vocab_file, 'r', encoding='utf-8') as f_v:
        inter_vocab = json.load(f_v)
    # transfer by dict
    for k in inter_vocab:
        inter_vocab[k] = transfer_dict[str(inter_vocab[k])]
    inter_vocab = list(inter_vocab.items())
    inter_vocab = sorted(inter_vocab, key=lambda x : x[1])
    tokens = [e[0] for e in inter_vocab]

    tail = [fairseq_vocab[-4],
            fairseq_vocab[-3],
            fairseq_vocab[-2],
            fairseq_vocab[-1]]
    assert tail == ['madeupword0000',
                    'madeupword0001',
                    'madeupword0002',
                    '<mask>']
    all_tokens = ['<s>', '<pad>', '</s>', '<unk>'] + \
                 tokens + tail

    gluon_vocab = gluon_Vocab(all_tokens,
                        unk_token=fairseq_vocab.unk_word,
                        pad_token=fairseq_vocab.pad_word,
                        eos_token=fairseq_vocab.eos_word,
                        bos_token=fairseq_vocab.bos_word,
                        mask_token=fairseq_vocab[-1])
    gluon_vocab.save(vocab_save_path)
    os.remove(temp_vocab_file)
    os.remove(temp_merges_file)

    gluon_tokenizer = HuggingFaceByteBPETokenizer(
        merges_save_path,
        vocab_save_path
    )

    if args.test:
        test_vocab(fairseq_model, gluon_tokenizer)

    vocab_size = len(fairseq_vocab)
    print('| converted dictionary: {} types'.format(vocab_size))
    return vocab_size

def test_vocab(fairseq_model, gluon_tokenizer, check_all_tokens=False):
    print('testing vocab')
    fairseq_vocab = fairseq_model.task.dictionary
    gluon_vocab = gluon_tokenizer.vocab
    assert len(fairseq_vocab) == \
           len(gluon_vocab)

    # assert all_tokens
    # roberta with gpt2 bytebpe bpe does not provide all tokens directly
    if check_all_tokens:
        for i in range(len(fairseq_vocab)):
            assert fairseq_vocab[i] == gluon_vocab.all_tokens[i], \
                   '{}, {}, {}'.format(i, fairseq_vocab[i], gluon_vocab.all_tokens[i])

    # assert special tokens
    for special_tokens in ['unk', 'pad', 'eos', 'bos']:
        assert getattr(fairseq_vocab, special_tokens + '_index') == \
               getattr(gluon_vocab, special_tokens + '_id')
        assert getattr(fairseq_vocab, special_tokens + '_word') == \
               getattr(gluon_vocab, special_tokens + '_token')
        # <mask> is the last one
        assert fairseq_vocab[-1] == \
               gluon_vocab.all_tokens[-1] == \
               '<mask>'

    sentence = "Hello, y'all! How are you Ⅷ 😁 😁 😁 ?" + \
               'GluonNLP is great！！！!!!' + \
               "GluonNLP-Amazon-Haibin-Leonard-Sheng-Shuai-Xingjian...../:!@# 'abc'"
    # assert encode
    fs_tokens = fairseq_model.encode(sentence)
    gl_tokens = gluon_tokenizer.encode(sentence, int)
    # Notice: we may append bos and eos
    # manuually after tokenizing sentences
    assert fs_tokens.numpy().tolist()[1:-1] == gl_tokens

    # assert decode
    fs_sentence = fairseq_model.decode(fs_tokens)
    gl_sentence = gluon_tokenizer.decode(gl_tokens)
    assert fs_sentence == gl_sentence

def convert_config(fairseq_cfg, vocab_size, cfg):
    print('converting config')
    cfg.defrost()
    cfg.MODEL.vocab_size = vocab_size
    cfg.MODEL.units = fairseq_cfg.encoder_embed_dim
    cfg.MODEL.hidden_size = fairseq_cfg.encoder_ffn_embed_dim
    cfg.MODEL.max_length = fairseq_cfg.max_positions
    cfg.MODEL.num_heads = fairseq_cfg.encoder_attention_heads
    cfg.MODEL.num_layers = fairseq_cfg.encoder_layers
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.activation = fairseq_cfg.activation_fn
    cfg.MODEL.pooler_activation = fairseq_cfg.pooler_activation_fn
    cfg.MODEL.layer_norm_eps = 1E-5
    cfg.MODEL.hidden_dropout_prob = fairseq_cfg.dropout
    cfg.MODEL.attention_dropout_prob = fairseq_cfg.attention_dropout
    cfg.MODEL.dtype = 'float32'
    cfg.INITIALIZER.embed = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg

def convert_params(fairseq_model,
                   gluon_cfg,
                   gluon_model_cls,
                   ctx,
                   gluon_prefix='robert_'):
    print('converting params')
    fairseq_params = fairseq_model.state_dict()
    fairseq_prefix = 'model.decoder.'
    gluon_model = gluon_model_cls.from_cfg(
        gluon_cfg,
        use_mlm=True,
        use_pooler=False,
        output_all_encodings=True,
        prefix=gluon_prefix
    )
    gluon_model.initialize(ctx=ctx)
    gluon_model.hybridize()
    gluon_params = gluon_model.collect_params()
    num_layers = gluon_cfg.MODEL.num_layers
    for layer_id in range(num_layers):
        fs_atten_prefix = \
            '{}sentence_encoder.layers.{}.self_attn.' \
            .format(fairseq_prefix, layer_id)
        fs_q_weight = fairseq_params[fs_atten_prefix + 'q_proj.weight'].cpu().numpy()
        fs_k_weight = fairseq_params[fs_atten_prefix + 'k_proj.weight'].cpu().numpy()
        fs_v_weight = fairseq_params[fs_atten_prefix + 'v_proj.weight'].cpu().numpy()
        fs_q_bias = fairseq_params[fs_atten_prefix + 'q_proj.bias'].cpu().numpy()
        fs_k_bias = fairseq_params[fs_atten_prefix + 'k_proj.bias'].cpu().numpy()
        fs_v_bias = fairseq_params[fs_atten_prefix + 'v_proj.bias'].cpu().numpy()
        gl_qkv_prefix = \
            '{}encoder_layers_{}_attn_qkv_' \
            .format(gluon_prefix, layer_id)
        gl_qkv_weight = gluon_params[gl_qkv_prefix + 'weight']
        gl_qkv_bias = gluon_params[gl_qkv_prefix + 'bias']
        gl_qkv_weight.set_data(
            np.concatenate([fs_q_weight, fs_k_weight, fs_v_weight], axis=0))
        gl_qkv_bias.set_data(
            np.concatenate([fs_q_bias, fs_k_bias, fs_v_bias], axis=0))

        for k, v in [
            ('self_attn.out_proj.weight', 'proj_weight'),
            ('self_attn.out_proj.bias', 'proj_bias'),
            ('self_attn_layer_norm.weight', 'ln_gamma'),
            ('self_attn_layer_norm.bias', 'ln_beta'),
            ('fc1.weight', 'ffn_ffn1_weight'),
            ('fc1.bias', 'ffn_ffn1_bias'),
            ('fc2.weight', 'ffn_ffn2_weight'),
            ('fc2.bias', 'ffn_ffn2_bias'),
            ('final_layer_norm.weight', 'ffn_ln_gamma'),
            ('final_layer_norm.bias', 'ffn_ln_beta')
        ]:
            fs_name = '{}sentence_encoder.layers.{}.{}' \
                      .format(fairseq_prefix, layer_id, k)
            gl_name = '{}encoder_layers_{}_{}' \
                      .format(gluon_prefix, layer_id, v)
            gluon_params[gl_name].set_data(
                fairseq_params[fs_name].cpu().numpy())

    for k, v in [
        ('sentence_encoder.embed_tokens.weight', 'tokens_embed_weight'),
        ('sentence_encoder.emb_layer_norm.weight', 'embed_ln_gamma'),
        ('sentence_encoder.emb_layer_norm.bias', 'embed_ln_beta'),
        ('lm_head.dense.weight', 'lm_dense1_weight'),
        ('lm_head.dense.bias', 'lm_dense1_bias'),
        ('lm_head.layer_norm.weight', 'lm_ln_gamma'),
        ('lm_head.layer_norm.bias', 'lm_ln_beta'),
        ('lm_head.bias', 'tokens_embed_bias')
    ]:
        fs_name = fairseq_prefix + k
        gl_name = gluon_prefix + v
        gluon_params[gl_name].set_data(
            fairseq_params[fs_name].cpu().numpy())

    # position embed weight
    padding_idx = fairseq_model.task.dictionary.pad_index
    fs_pos_embed_name = fairseq_prefix + 'sentence_encoder.embed_positions.weight'
    gl_pos_embed_name = gluon_prefix + 'pos_embed_embed_weight'
    gluon_params[gl_pos_embed_name].set_data(
        fairseq_params[fs_pos_embed_name].cpu().numpy()[padding_idx + 1:,:])

    # assert untie=False
    assert np.array_equal(
        fairseq_params[fairseq_prefix + 'sentence_encoder.embed_tokens.weight'].cpu().numpy(),
        fairseq_params[fairseq_prefix + 'lm_head.weight'].cpu().numpy()
    )

    return gluon_model

def test_model(fairseq_model, gluon_model, gpu):
    print('testing model')
    ctx = mx.gpu(gpu) if gpu is not None else mx.cpu()
    batch_size = 3
    seq_length = 32
    vocab_size = len(fairseq_model.task.dictionary)
    padding_id = fairseq_model.model.decoder.sentence_encoder.padding_idx
    input_ids = np.random.randint( # skip padding_id
        padding_id + 1,
        vocab_size,
        (batch_size, seq_length)
    )
    valid_length = np.random.randint(
        seq_length // 2,
        seq_length,
        (batch_size,)
    )
    for i in range(batch_size): # add padding, for fairseq padding mask
        input_ids[i,valid_length[i]:] = padding_id

    gl_input_ids = mx.np.array(input_ids, dtype=np.int32, ctx=ctx)
    gl_valid_length = mx.np.array(valid_length, dtype=np.int32, ctx=ctx)

    fs_input_ids = torch.from_numpy(input_ids).cuda(gpu)
    if gpu is not None:
        fs_input_ids = fs_input_ids.cuda(gpu)

    fairseq_model.model.eval()

    gl_all_hiddens, gl_x = \
        gluon_model(gl_input_ids, gl_valid_length)

    fs_x, fs_extra = \
        fairseq_model.model.cuda(gpu)(fs_input_ids, return_all_hiddens=True)
    fs_all_hiddens = fs_extra['inner_states']

    num_layers = fairseq_model.args.encoder_layers
    for i in range(num_layers + 1):
        gl_hidden = gl_all_hiddens[i].asnumpy()
        fs_hidden = fs_all_hiddens[i]
        fs_hidden = fs_hidden.transpose(0, 1)
        fs_hidden = fs_hidden.detach().cpu().numpy()
        for j in range(batch_size):
            assert_allclose(
                gl_hidden[j, :valid_length[j], :],
                fs_hidden[j, :valid_length[j], :],
                1E-3,
                1E-3
            )

    gl_x = gl_x.asnumpy()
    fs_x = fs_x.transpose(0, 1)
    fs_x = fs_x.detach().cpu().numpy()
    for j in range(batch_size):
        assert_allclose(
            gl_x[j, :valid_length[j], :],
            fs_x[j, :valid_length[j], :],
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

    fairseq_roberta = fairseq_RobertaModel.from_pretrained(args.fairseq_model_path,
                                                           checkpoint_file='model.pt')
    vocab_size = convert_vocab(args, fairseq_roberta)

    gluon_cfg = convert_config(fairseq_roberta.args, vocab_size,
                               gluon_RobertaModel.get_cfg().clone())
    with open(os.path.join(args.save_dir, 'model.yml'), 'w') as of:
        of.write(gluon_cfg.dump())

    ctx = mx.gpu(args.gpu) if args.gpu is not None else mx.cpu()
    gluon_roberta = convert_params(fairseq_roberta,
                                   gluon_cfg,
                                   gluon_RobertaModel,
                                   ctx,
                                   gluon_prefix='roberta_')

    if args.test:
        test_model(fairseq_roberta, gluon_roberta, args.gpu)

    gluon_roberta.save_parameters(os.path.join(args.save_dir, 'model.params'), deduplicate=True)
    logging.info('Convert the RoBERTa model in {} to {}'.
                 format(os.path.join(args.fairseq_model_path, 'model.pt'), \
                        os.path.join(args.save_dir, 'model.params')))
    logging.info('Conversion finished!')
    logging.info('Statistics:')
    rename(args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    logging_config()
    convert_fairseq_model(args)
