import os
import pytest
from gluonnlp.cli.process import learn_subword, apply_subword
from gluonnlp.data import tokenizers

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


def get_test_pair_corpus(key='de-en'):
    if key == 'de-en':
        return [os.path.join(_CURR_DIR, 'data', 'wmt19-test-de-en.de'),
                os.path.join(_CURR_DIR, 'data', 'wmt19-test-de-en.en')]
    elif key == 'zh-en':
        return [os.path.join(_CURR_DIR, 'data', 'wmt19-test-zh-en.zh'),
                os.path.join(_CURR_DIR, 'data', 'wmt19-test-zh-en.en')]
    else:
        raise NotImplementedError


@pytest.mark.parametrize('model',
                         ['yttm', 'spm', 'subword_nmt', 'hf_bpe', 'hf_bytebpe', 'hf_wordpiece'])
def test_subword_algorithms_ende(model):
    dir_path = os.path.join(_CURR_DIR, 'learn_apply_subword_ende_results')
    os.makedirs(dir_path, exist_ok=True)
    dir_path = os.path.realpath(dir_path)
    parser = learn_subword.get_parser()
    apply_parser = apply_subword.get_parser()
    corpus_path_pair = [os.path.join(_CURR_DIR, 'data', 'wmt19-test-de-en.de'),
                        os.path.join(_CURR_DIR, 'data', 'wmt19-test-de-en.en')]
    args = parser.parse_args(['--corpus'] + corpus_path_pair +
                             ['--model', model, '--vocab-size', '5000',
                              '--save-dir', dir_path])
    # Train the tokenizer
    learn_subword.main(args)
    if model in ['yttm', 'spm', 'subword_nmt']:
        model_key = model
    else:
        model_key = 'hf_tokenizer'
    tokenizer = tokenizers.create(model_key,
                                  model_path=os.path.join(dir_path,
                                                          '{}.model'.format(model)),
                                  vocab=os.path.join(dir_path,
                                                     '{}.vocab'.format(model)))
    args = apply_parser.parse_args(['--corpus'] + [corpus_path_pair[0]] +
                                   ['--model', model,
                                    '--model-path', os.path.join(dir_path,
                                                                 '{}.model'.format(model)),
                                    '--vocab-path', os.path.join(dir_path,
                                                                 '{}.vocab'.format(model)),
                                    '--save-path',
                                    os.path.join(dir_path,
                                                 'wmt19-test-de-en.de.{}'.format(model))])
    apply_subword.main(args)
    args = apply_parser.parse_args(['--corpus'] + [corpus_path_pair[1]] +
                                   ['--model', model,
                                    '--model-path', os.path.join(dir_path,
                                                                 '{}.model'.format(model)),
                                    '--vocab-path', os.path.join(dir_path,
                                                                 '{}.vocab'.format(model)),
                                    '--save-path',
                                    os.path.join(dir_path,
                                                 'wmt19-test-de-en.en.{}'.format(model))])
    apply_subword.main(args)

    # Decode back with the trained tokenizer
    for prefix_fname in ['wmt19-test-de-en.de.{}'.format(model),
                         'wmt19-test-de-en.en.{}'.format(model)]:
        with open(os.path.join(dir_path, '{}.decode'.format(prefix_fname)),
                  'w', encoding='utf-8') as out_f:
            with open(os.path.join(dir_path, '{}'.format(prefix_fname)), 'r',
                      encoding='utf-8') as in_f:
                for line in in_f:
                    out_f.write(tokenizer.decode(line.split()) + '\n')


@pytest.mark.parametrize('model',
                         ['yttm', 'spm', 'subword_nmt', 'hf_bpe', 'hf_bytebpe', 'hf_wordpiece'])
def test_subword_algorithms_zh(model):
    dir_path = os.path.join(_CURR_DIR, 'learn_apply_subword_zh_results')
    os.makedirs(dir_path, exist_ok=True)
    dir_path = os.path.realpath(dir_path)
    parser = learn_subword.get_parser()
    apply_parser = apply_subword.get_parser()
    corpus_path = os.path.join(_CURR_DIR, 'data', 'wmt19-test-zh-en.zh.jieba')
    arguments = ['--corpus'] + [corpus_path] +\
                ['--model', model, '--vocab-size', '5000', '--save-dir', dir_path]
    args = parser.parse_args(arguments)
    # Train the tokenizer
    learn_subword.main(args)
    if model in ['yttm', 'spm', 'subword_nmt']:
        model_key = model
    else:
        model_key = 'hf_tokenizer'
    tokenizer = tokenizers.create(model_key,
                                  model_path=os.path.join(dir_path,
                                                          '{}.model'.format(model)),
                                  vocab=os.path.join(dir_path,
                                                     '{}.vocab'.format(model)))
    arguments = ['--corpus'] + [corpus_path] +\
                ['--model', model,
                 '--model-path', os.path.join(dir_path, '{}.model'.format(model)),
                 '--vocab-path', os.path.join(dir_path, '{}.vocab'.format(model)),
                 '--save-path', os.path.join(dir_path,
                                             'wmt19-test-zh-en.zh.jieba.{}'.format(model))]
    args = apply_parser.parse_args(arguments)
    apply_subword.main(args)

    # Decode back with the trained tokenizer
    for prefix_fname in ['wmt19-test-zh-en.zh.jieba.{}'.format(model)]:
        with open(os.path.join(dir_path, '{}.decode'.format(prefix_fname)),
                  'w', encoding='utf-8') as out_f:
            with open(os.path.join(dir_path, '{}'.format(prefix_fname)),
                      'r', encoding='utf-8') as in_f:
                for line in in_f:
                    out_f.write(tokenizer.decode(line.split()) + '\n')
