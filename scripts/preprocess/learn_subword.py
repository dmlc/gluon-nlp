from gluonnlp.utils.lazy_imports import try_import_sentencepiece,\
    try_import_subword_nmt, try_import_yttm, try_import_huggingface_tokenizers
import argparse
import textwrap
import os
from collections import OrderedDict
import json
from uuid import uuid4
from gluonnlp.data import Vocab


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
    Learn BPE based on different implementations.

    We support the following models:

        "python3 learn_subword.py --model spm" : Train a Sentencepiece Model on raw text;
        "python3 learn_subword.py --model subword_nmt" : Train with the subword-nmt package;
        "python3 learn_subword.py --model yttm" : Train with YouTokenToMe; 
        "python3 learn_subword.py --model hf_bytebpe" : Train with the Byte-level BPE Tokenizer Implemented by Huggingface.
        "python3 learn_subword.py --model hf_wordpiece" : Train with the Wordpiece Tokenizer Implementated by Huggingface.
        "python3 learn_subword.py --model hf_bpe" : Train with the BPE Tokenizer Implemented by Huggingface.
    ''')
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
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory for saving the model and vocabulary file')
    parser.add_argument('--coverage', type=float, default=1.0, 
                        help='Amount of characters covered by the model, '
                             'this is only applicable to spm and yttm')
    parser.add_argument('--n-threads', type=int, default=-1,
                        help='Number of threads, only applicable to yttm')
    parser.add_argument('--input-sentence-size', type=int, default=1000000,
                        help='Size of input sentence, only applicable to sentencepiece, '
                        'you can reduce this value when getting out of memory error')
    parser.add_argument('--lowercase', action='store_true', default=False,
                        help='Use lowercase, '
                        'only applicable to hf_bpe, hf_bytebpe and hf_wordpiece')
    parser.add_argument('--strip-accents', action='store_true', default=False,
                        help='Disable BERT characters normalization, '
                        'only applicable to hf_wordpiece')
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
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_prefix = os.path.join(args.save_dir, args.model)
    special_tokens_kv = OrderedDict()
    # unk is always required
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
    # hf_wordpiece must contains mask, cls and sep tokens
    # the costom defined mask,cls,sep can overwrite the default settings
    if args.model == 'hf_wordpiece':
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
            while (pair):
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
            tokenizer = tokenizers.CharBPETokenizer(lowercase=args.lowercase)
        elif args.model == 'hf_bytebpe':
            tokenizer = tokenizers.ByteLevelBPETokenizer(lowercase=args.lowercase)
        elif args.model == 'hf_wordpiece':
            tokenizer = tokenizers.BertWordPieceTokenizer(lowercase=args.lowercase,
                                                          strip_accents=args.strip_accents)
        else:
            raise NotImplementedError
        tokenizer.train(
            corpus_path_list,
            vocab_size=args.vocab_size,
            show_progress=True,
            special_tokens=special_tokens)
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
    unk_token = special_tokens_kv.pop('unk_token')
    vocab_obj = Vocab(vocab, unk_token=unk_token, **special_tokens_kv)
    vocab_obj.save(model_prefix + '.vocab')


def cat_corpus(corpus_path_list):
    # TODO Use temporary file
    corpus_path = "./" + str(uuid4()) + '.corpus'
    with open(corpus_path, 'wb') as cat_corpus:
        for cp in corpus_path_list:
            with open(cp, 'rb') as corpus:
                cat_corpus.write(corpus.read())
    return corpus_path


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
