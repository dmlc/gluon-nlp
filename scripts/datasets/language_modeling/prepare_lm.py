import argparse
import os
import zipfile
from typing import List, Optional
from collections import Counter
from gluonnlp.base import get_data_home_dir
from gluonnlp.utils.misc import download
from gluonnlp.data.vocab import Vocab


_CITATIONS = """
@ONLINE {mahoney2011large,
  title={Large text compression benchmark},
  author={Mahoney, Matt},
  url={http://www.mattmahoney.net/text/text.html},
  year={2011}
}

@article{chelba2013one,
  title={One billion word benchmark for measuring progress in statistical language modeling},
  author={Chelba, Ciprian and Mikolov, Tomas and Schuster, Mike and Ge, Qi and Brants, Thorsten
   and Koehn, Phillipp and Robinson, Tony},
  journal={arXiv preprint arXiv:1312.3005},
  year={2013}
}


@inproceedings{merity2016pointer,
  title={Pointer sentinel mixture models},
  author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  booktitle={ICLR},
  year={2017}
}
"""

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums',
                                    'language_model.txt')
_URL_FILE_STATS = dict()
for line in open(_URL_FILE_STATS_PATH, 'r', encoding='utf-8'):
    url, hex_hash, file_size = line.strip().split()
    _URL_FILE_STATS[url] = hex_hash

_URLS = {
    'wikitext2': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    'wikitext103': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    'enwik8': 'http://mattmahoney.net/dc/enwik8.zip',
    'text8': 'http://mattmahoney.net/dc/text8.zip',
    'gbw': 'http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz'
}


def get_parser():
    parser = argparse.ArgumentParser(description='Downloading and Preprocessing'
                                                 ' Language Modeling Datasets.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wikitext2', 'wikitext103', 'text8', 'enwik8', 'gbw'],
                        help='The dataset to use.')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='The directory to save the dataset.'
                             ' By default, it will save to a folder with the same name as the '
                             'dataset')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite the saved '
                                                                 'files.')
    parser.add_argument('--cache-path', type=str,
                        default=os.path.join(get_data_home_dir(), 'lm_benchmark_data'),
                        help='The temporary path to download the dataset.')
    return parser


def path_exist_and_skip(path, overwrite):
    if os.path.exists(path) and not overwrite:
        print('Found {}. Skip writing. Turn `--overwrite` to force update the file.'
              .format(path))
        return True
    return False


def build_vocab(corpus_path_l: List, eos_token: Optional[str] = '<eos>') -> Vocab:
    """Build the default vocabulary used in datasets like

        - wikitext2
        - wikitext103
        - text8
        - enwiki8

    The strategy is to split with white-space and store all appeared tokens.
    Also, the tokens will be sorted with a descending order of their frequency.

    Parameters
    ----------
    corpus_path_l
        The corpus path
    eos_token
        If it is not None, the eos_token will be added to the vocabulary.

    Returns
    -------
    vocab
        The vocabulary
    """
    counter = Counter()
    ntokens = 0
    print('Build the default vocabulary used in benchmarks:')
    for corpus_path in corpus_path_l:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                line = line.strip()
                tokens = line.split()
                counter.update(tokens)
                ntokens += len(tokens)
    if eos_token is not None and eos_token in counter:
        raise ValueError('eos_token is set to be "{}", which appears in the text. '
                         'Is it intended? You may choose another token as the eos_token.'
                         .format(eos_token))
    vocab = Vocab(counter, unk_token=None, eos_token=eos_token)
    print('Processed {} tokens, vocab={}'.format(ntokens, vocab))
    return vocab


def main(args):
    # Download the data
    url = _URLS[args.dataset]
    file_hash = _URL_FILE_STATS[url]
    target_download_location = os.path.join(args.cache_path,
                                            os.path.basename(url))
    download(url, target_download_location, sha1_hash=file_hash)
    save_dir = args.dataset if args.save_dir is None else args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Extract and process the data
    if args.dataset == 'wikitext2':
        with zipfile.ZipFile(target_download_location) as zf:
            train_data = zf.read('wikitext-2/wiki.train.tokens')
            valid_data = zf.read('wikitext-2/wiki.valid.tokens')
            test_data = zf.read('wikitext-2/wiki.test.tokens')
            for filename, part in [('train.txt', train_data),
                                   ('valid.txt', valid_data),
                                   ('test.txt', test_data)]:
                filename = os.path.join(save_dir, filename)
                print('{} will have {} bytes'.format(filename, len(part)))
                if not path_exist_and_skip(filename, args.overwrite):
                    with open(filename, 'wb') as of:
                        of.write(part)
            vocab = build_vocab([os.path.join(save_dir, 'train.txt'),
                                 os.path.join(save_dir, 'valid.txt'),
                                 os.path.join(save_dir, 'test.txt')])
            vocab.save(os.path.join(save_dir, 'vocab.json'))
    elif args.dataset == 'wikitext103':
        with zipfile.ZipFile(target_download_location) as zf:
            train_data = zf.read('wikitext-103/wiki.train.tokens')
            valid_data = zf.read('wikitext-103/wiki.valid.tokens')
            test_data = zf.read('wikitext-103/wiki.test.tokens')
            for filename, part in [('train.txt', train_data),
                                   ('valid.txt', valid_data),
                                   ('test.txt', test_data)]:
                filename = os.path.join(save_dir, filename)
                if not path_exist_and_skip(filename, args.overwrite):
                    print('{} will have {} bytes'.format(filename, len(part)))
                    with open(filename, 'wb') as of:
                        of.write(part)
            vocab = build_vocab([os.path.join(save_dir, 'train.txt')])
            vocab.save(os.path.join(save_dir, 'vocab.json'))
    elif args.dataset == 'text8':
        with zipfile.ZipFile(target_download_location) as zf:
            with zf.open('text8', 'r') as f:
                data = f.read().decode('utf-8')
                num_test_chars = 5000000
                train_data = data[: -2 * num_test_chars]
                valid_data = data[-2 * num_test_chars: -num_test_chars]
                test_data = data[-num_test_chars:]
                for filename, part in [('train.txt', train_data),
                                       ('valid.txt', valid_data),
                                       ('test.txt', test_data)]:
                    filename = os.path.join(save_dir, filename)
                    print('{} will have {} bytes'.format(filename, len(part)))
                    print('- Tokenizing...')
                    # Change space ' ' to underscore '_'
                    part_str = ' '.join(['_' if c == ' ' else c for c in part.strip()])
                    print('- Writing...')
                    if not path_exist_and_skip(filename, args.overwrite):
                        with open(filename, 'w', encoding='utf-8') as of:
                            of.write(part_str)
                    if not path_exist_and_skip(filename + '.raw', args.overwrite):
                        with open(filename + '.raw', 'w', encoding='utf-8') as of:
                            of.write(part)
            vocab = build_vocab([os.path.join(save_dir, 'train.txt')], eos_token=None)
            vocab.save(os.path.join(save_dir, 'vocab.json'))
    elif args.dataset == 'enwik8':
        with zipfile.ZipFile(target_download_location) as zf:
            data = zf.read('enwik8')
            print('Length of enwik8: {}'.format(len(data)))
            num_test_chars = 5000000
            train_data = data[: -2 * num_test_chars]
            valid_data = data[-2 * num_test_chars: -num_test_chars]
            test_data = data[-num_test_chars:]

            for filename, part in [('train.txt', train_data),
                                   ('valid.txt', valid_data),
                                   ('test.txt', test_data)]:
                filename = os.path.join(save_dir, filename)
                print('{} will have {} bytes'.format(filename, len(part)))
                print('- Tokenizing...')
                part_str = ' '.join([str(c) if c != ord('\n') else '\n' for c in part])
                print('- Writing...')
                if not path_exist_and_skip(filename, args.overwrite):
                    with open(filename, 'w') as of:
                        of.write(part_str)
                if not path_exist_and_skip(filename + '.raw', args.overwrite):
                    with open(filename + '.raw', 'wb') as of:
                        of.write(part)
            vocab = build_vocab([os.path.join(save_dir, 'train.txt')], eos_token=None)
            vocab.save(os.path.join(save_dir, 'vocab.json'))

    elif args.dataset == 'gbw':
        pass
    else:
        raise NotImplementedError


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
