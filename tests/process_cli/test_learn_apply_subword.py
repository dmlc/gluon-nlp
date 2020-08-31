import os
import tempfile
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


def verify_subword_algorithms_ende(dir_path):
    parser = learn_subword.get_parser()
    apply_parser = apply_subword.get_parser()
    corpus_path_pair = [os.path.join(_CURR_DIR, 'data', 'wmt19-test-de-en.de'),
                        os.path.join(_CURR_DIR, 'data', 'wmt19-test-de-en.en')]
    for model in ['yttm', 'spm', 'subword_nmt', 'hf_bpe', 'hf_bytebpe', 'hf_wordpiece']:
        args = parser.parse_args(['--corpus'] + corpus_path_pair +
                                 ['--model', model, '--vocab-size', 5000,
                                  '--save-dir', dir_path])
        # Train the tokenizer
        learn_subword.main(args)
        args = apply_parser.parse_args(['--corpus'] + corpus_path_pair[0] +
                                       ['--model', model,
                                        '--save-path',
                                        os.path.join(dir_path,
                                                     'wmt19-test-de-en.de.{}'.format(model))])
        apply_subword.main(args)


if __name__ == '__main__':
    verify_subword_algorithms_ende('.')
