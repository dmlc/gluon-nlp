from gluonnlp.utils.lazy_imports import try_import_sentencepiece,\
    try_import_subword_nmt, try_import_yttm, try_import_huggingface_tokenizers
from pkg_resources import parse_version
import argparse
import warnings
import textwrap
import os
from collections import OrderedDict
import json
from uuid import uuid4
from packaging import version
from gluonnlp.data import Vocab


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
    Learn BPE based on different implementations.

    We support the following models:

        - "nlp_process learn_subword --model spm --corpus CORPUS --vocab-size SIZE"
            Train a Sentencepiece Model on raw text.

        - "nlp_process learn_subword --model subword_nmt --corpus CORPUS --vocab-size SIZE"
            Train with the subword-nmt package:

        - "nlp_process learn_subword --model yttm --corpus CORPUS --vocab-size SIZE"
            Train with YouTokenToMe:

        - "nlp_process learn_subword --model hf_bytebpe --corpus CORPUS --vocab-size SIZE"
            Train with the Byte-level BPE Tokenizer Implemented by Huggingface.

        - "nlp_process learn_subword --model hf_wordpiece --corpus CORPUS --vocab-size SIZE"
            Train with the Wordpiece Tokenizer Implementated by Huggingface.

        - "nlp_process learn_subword --model hf_bpe --corpus CORPUS --vocab-size SIZE"
            Train with the BPE Tokenizer Implemented by Huggingface.
    '''),
        prog='learn_subword'
    )
    parser.add_argument('--corpus', type=str, nargs='+', required=True,
                        help='Path of the corpus. '
                             'You may input multiple corpus files separated by space.')
    parser.add_argument('--vocab-size', type=int, required=True,
                        help='Estimated learned vocabulary size')
    parser.add_argument('--model', type=str, choices=['spm',
                                                      'subword_nmt',
                                                      'yttm',
                                                      'hf_bytebpe',
                                                      'hf_wordpiece',
                                                      'hf_bpe'],
                        required=True, help='Subword model type')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory for saving the model and vocabulary file')
    parser.add_argument('--coverage', type=float, default=1.0, 
                        help='Amount of characters covered by the model, '
                             'this is only applicable to spm and yttm')
    parser.add_argument('--n-threads', type=int, default=-1,
                        help='Number of threads, only applicable to yttm')
    parser.add_argument('--input-sentence-size', type=int, default=1000000,
                        help='Size of input sentence. Internally, the algorithm will randomly '
                             'sample some sentences to train the tokenizer. '
                             'This is only applicable to sentencepiece, '
                             'you can reduce this value when getting out of memory error.')
    parser.add_argument('--lowercase', action='store_true', default=False,
                        help='Use lowercase, '
                        'only applicable to hf_bpe, hf_bytebpe and hf_wordpiece')
    parser.add_argument('--bert-normalizer', action='store_true', default=False,
                        help='Whether to use the Normalizer in BERT. '
                             'This will be used only when you choose the huggingface models. '
                             'Basically, the BERT Normalizer will '
                             '1) remove control characters. '
                             '2) putting spaces around chinese characters. '
                             '3) strip accents. '
                             'For more details, you can refer to '
                             'https://github.com/google-research/bert/'
                             'blob/master/multilingual.md#tokenization')
    parser.add_argument('--split-punctuation', action='store_true', default=False,
                        help='Whether to split on punctuation.')
    parser.add_argument('--disable-unk', action='store_true', default=False,
                        help='Whether to disable unk token (default settings enable unk)')
    parser.add_argument('--disable-bos', action='store_true', default=False,
                        help='Disable bos token (default settings enable bos)')
    parser.add_argument('--disable-eos', action='store_true', default=False,
                        help='Disable eos token (default settings enable eos)')
    parser.add_argument('--disable-pad', action='store_true', default=False,
                        help='Disable pad token (default settings enable pad)')
    parser.add_argument('--custom-special-tokens', type=str, nargs='*', default=[], 
                        help='Specified special tokens key value pairs besides unk, '
                             'bos, eos and pad, for example: '
                             '--custom special tokens cls_token=<cls> sep_token=<sep>, '
                             'this is not applicable to yttm')
    return parser


def main(args):
    corpus_path_list = args.corpus
    if args.save_dir is None:
        args.save_dir = args.model
    for corpus_path in corpus_path_list:
        if not os.path.exists(corpus_path):
            raise ValueError('The path="{}" provided by --corpus does not exist!'
                             .format(corpus_path))
    print('Learn the {} subword model based on {}.'.format(args.model, args.corpus))
    os.makedirs(args.save_dir, exist_ok=True)
    model_prefix = os.path.join(args.save_dir, args.model)
    special_tokens_kv = OrderedDict()
    if not args.disable_unk:
        special_tokens_kv['unk_token'] = Vocab.UNK_TOKEN
    if not args.disable_bos:
        special_tokens_kv['bos_token'] = Vocab.BOS_TOKEN
    if not args.disable_eos:
        special_tokens_kv['eos_token'] = Vocab.EOS_TOKEN
    if not args.disable_pad:
        special_tokens_kv['pad_token'] = Vocab.PAD_TOKEN
    # split custom special tokens
    if args.model in ['yttm'] and len(args.custom_special_tokens) > 0:
        raise ValueError('model {} do not support custom_special_tokens'.format(args.model))
    for custom_special_token in args.custom_special_tokens:
        kv = custom_special_token.split('=')
        if not len(kv) == 2:
            raise ValueError('parameter {} has wrong format'.format(custom_special_token))
        k, v = kv[0], kv[1]
        if k in special_tokens_kv:
            raise ValueError('There are overlaps between the custom special tokens and the'
                             ' unk, bos, eos, pad tokens')
        special_tokens_kv[k] = v
    if args.model == 'hf_wordpiece':
        tokenizers = try_import_huggingface_tokenizers()
        if 'unk_token' not in special_tokens_kv or special_tokens_kv['unk_token'] != '[UNK]':
            # TODO, HF Tokenizer must have the unk token.
            special_tokens_kv['unk_token'] = '[UNK]'
        if parse_version(tokenizers.__version__) < parse_version('0.8'):
            # The older version of Tokenizers
            # hf_wordpiece must contain mask, cls and sep tokens
            # the custom defined mask,cls,sep can overwrite the default settings
            if 'mask_token' not in special_tokens_kv:
                special_tokens_kv['mask_token'] = Vocab.MASK_TOKEN
            if 'cls_token' not in special_tokens_kv:
                special_tokens_kv['cls_token'] = Vocab.CLS_TOKEN
            if 'sep_token' not in special_tokens_kv:
                special_tokens_kv['sep_token'] = Vocab.SEP_TOKEN
    special_tokens = list(special_tokens_kv.values())
    print('special tokens: ' + ', '.join(special_tokens))
    vocab = []
    if args.model == 'spm':
        try_import_sentencepiece()
        import sentencepiece as spm
        corpus_path = ','.join(corpus_path_list)
        script = '--input={} --model_prefix={} --vocab_size={} --character_coverage={} --input_sentence_size={}' \
                 .format(corpus_path, model_prefix, args.vocab_size, args.coverage, args.input_sentence_size)
        script += (' --unk_id=' + str(special_tokens.index(Vocab.UNK_TOKEN)))
        script += (' --bos_id=' + ('-1' if args.disable_bos else str(special_tokens.index(Vocab.BOS_TOKEN))))
        script += (' --eos_id=' + ('-1' if args.disable_eos else str(special_tokens.index(Vocab.EOS_TOKEN))))
        script += (' --pad_id=' + ('-1' if args.disable_pad else str(special_tokens.index(Vocab.PAD_TOKEN))))
        if len(args.custom_special_tokens) > 0:
            ids_in_script = script.count('_id')
            script += (' --control_symbols=' + ','.join(special_tokens[ids_in_script:]))
        print(script)
        spm.SentencePieceTrainer.Train(script)
        if 'bos_token' in special_tokens_kv:
            special_tokens_kv['bos_token'] = '<s>'
        if 'eos_token' in special_tokens_kv:
            special_tokens_kv['eos_token'] = '</s>'
        # build spm vocab
        spm_model = spm.SentencePieceProcessor()
        spm_model.load(model_prefix + '.model')
        vocab = [spm_model.id_to_piece(i) for i in range(len(spm_model))]
        os.remove(model_prefix + '.vocab')
    elif args.model == 'subword_nmt':
        try_import_subword_nmt()
        from subword_nmt import learn_bpe
        corpus_path = cat_corpus(corpus_path_list)\
            if len(corpus_path_list) > 1 else corpus_path_list[0]
        # build model
        with open(corpus_path, 'r', encoding='utf-8') as fc,\
             open(model_prefix + '.model', 'w', encoding='utf-8') as fm:
            learn_bpe.learn_bpe(fc, fm, args.vocab_size - len(special_tokens), total_symbols=True)
        # build vocab
        with open(corpus_path, 'r', encoding='utf-8') as fc, \
             open(model_prefix + '.model', 'r', encoding='utf-8') as fm:
            vocab.extend(special_tokens)
            uniq_chars_internal = set()
            uniq_chars_final = set()
            uniq_words = set()
            for line in fc:
                for word in line.strip('\r\n ').split(' '):
                    if word:
                        uniq_words.add(word)
            # this code piece is same as 
            # https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/learn_bpe.py shows
            uniq_words = [tuple(x[:-1]) + (x[-1]+'</w>',) for x in uniq_words]
            for word in uniq_words:
                for char in word[:-1]:
                    uniq_chars_internal.add(char)
                uniq_chars_final.add(word[-1])
            # sort to ensure the same settings produce the same vocab
            vocab.extend(sorted(list(uniq_chars_internal)))
            vocab.extend(sorted(list(uniq_chars_final)))
            fm.readline()
            pair = fm.readline()
            while pair:
                vocab.append(pair.replace(' ', '', 1).strip())
                pair = fm.readline()
        if len(corpus_path_list) > 1:
            os.remove(corpus_path)
    elif args.model == 'yttm':
        try_import_yttm()
        import youtokentome as yttm
        corpus_path = cat_corpus(corpus_path_list)\
            if len(corpus_path_list) > 1 else corpus_path_list[0]
        tokenizer = yttm.BPE.train(
            data=corpus_path, 
            model=model_prefix + '.model',
            vocab_size=args.vocab_size, 
            coverage=args.coverage, 
            n_threads=args.n_threads,
            unk_id=special_tokens.index(Vocab.UNK_TOKEN),
            bos_id=-1 if args.disable_bos else special_tokens.index(Vocab.BOS_TOKEN),
            eos_id=-1 if args.disable_eos else special_tokens.index(Vocab.EOS_TOKEN),
            pad_id=-1 if args.disable_pad else special_tokens.index(Vocab.PAD_TOKEN))
        vocab = tokenizer.vocab()
        if 'unk_token' in special_tokens_kv:
            special_tokens_kv['unk_token'] = '<UNK>'
        if 'bos_token' in special_tokens_kv:
            special_tokens_kv['bos_token'] = '<BOS>'
        if 'eos_token' in special_tokens_kv:
            special_tokens_kv['eos_token'] = '<EOS>'        
        if 'pad_token' in special_tokens_kv:
            special_tokens_kv['pad_token'] = '<PAD>'
        if len(corpus_path_list) > 1:
            os.remove(corpus_path)
    elif args.model in ['hf_bpe', 'hf_bytebpe', 'hf_wordpiece']:
        tokenizers = try_import_huggingface_tokenizers()
        if args.model == 'hf_bpe':
            split_on_whitespace_only = not args.split_punctuation
            tokenizer = tokenizers.CharBPETokenizer(
                lowercase=args.lowercase,
                bert_normalizer=args.bert_normalizer,
                split_on_whitespace_only=split_on_whitespace_only)
        elif args.model == 'hf_bytebpe':
            tokenizer = tokenizers.ByteLevelBPETokenizer(lowercase=args.lowercase)
        elif args.model == 'hf_wordpiece':
            unk_token = special_tokens_kv.get('unk_token', None)
            sep_token = special_tokens_kv.get('sep_token', None)
            cls_token = special_tokens_kv.get('cls_token', None)
            pad_token = special_tokens_kv.get('pad_token', None)
            mask_token = special_tokens_kv.get('mask_token', None)
            if args.bert_normalizer:
                strip_accents = None
                clean_text = True
                handle_chinese_chars = True
            else:
                strip_accents = False
                clean_text = False
                handle_chinese_chars = False
            tokenizer = tokenizers.BertWordPieceTokenizer(
                unk_token=unk_token,
                sep_token=sep_token,
                cls_token=cls_token,
                pad_token=pad_token,
                mask_token=mask_token,
                lowercase=args.lowercase,
                strip_accents=strip_accents,
                handle_chinese_chars=handle_chinese_chars,
                clean_text=clean_text
            )
        else:
            raise NotImplementedError
        tokenizer.train(
            corpus_path_list,
            vocab_size=args.vocab_size,
            show_progress=True,
            special_tokens=special_tokens)
        # Deal with the API change of tokenizers >= 0.8
        if version.parse(tokenizers.__version__) >= version.parse('0.8'):
            save_model_path = model_prefix + '.model'
            tokenizer.save(save_model_path)
            model_info = json.load(open(save_model_path, encoding='utf-8'))
            special_tokens_in_tokenizer = model_info['added_tokens']
            assert len(special_tokens_in_tokenizer) == len(special_tokens)
            hf_vocab = model_info['model']['vocab']
            hf_vocab_sorted = sorted(list(hf_vocab.items()), key=lambda x: x[1])
            hf_vocab_ids = [ele[1] for ele in hf_vocab_sorted]
            assert min(hf_vocab_ids) == 0 and max(hf_vocab_ids) == len(hf_vocab_ids) - 1
            vocab = [ele[0] for ele in hf_vocab_sorted]
        else:
            tokenizer.save(args.save_dir, args.model)
            # we replace the huggingface vocab file with our Vocab implementation
            if args.model == 'hf_wordpiece':
                hf_vocab_file = model_prefix + '-vocab.txt'
                with open(hf_vocab_file, 'r', encoding='utf-8') as fv:
                    for line in fv:
                        vocab.append(line.strip())
            else:
                # Move the hf_${model}-merges.txt to hf_${model}.models
                os.rename(os.path.join(args.save_dir, '{}-merges.txt'.format(args.model)),
                          os.path.join(args.save_dir, '{}.model'.format(args.model)))
                hf_vocab_file = model_prefix + '-vocab.json'
                with open(hf_vocab_file, 'r', encoding='utf-8') as fv:
                    vocab_kv = json.load(fv)
                    vocab_kv = sorted(list(vocab_kv.items()), key=lambda x: x[1])
                    for kv in vocab_kv:
                        vocab.append(kv[0])
            os.remove(hf_vocab_file)
    else:
        raise NotImplementedError
    vocab_obj = Vocab(vocab, **special_tokens_kv)
    vocab_obj.save(model_prefix + '.vocab')


def cat_corpus(corpus_path_list):
    # TODO Use temporary file
    corpus_path = "./" + str(uuid4()) + '.corpus'
    with open(corpus_path, 'wb') as of:
        for cp in corpus_path_list:
            with open(cp, 'rb') as corpus:
                of.write(corpus.read())
    return corpus_path


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
