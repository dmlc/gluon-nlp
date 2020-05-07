import glob
import os
import argparse
import zipfile
from gluonnlp.base import get_data_home_dir
from gluonnlp.utils.misc import download


_CITATIONS = r"""
@InProceedings{lahiri:2014:SRW,
  author    = {Lahiri, Shibamouli},
  title     = {{Complexity of Word Collocation Networks: A Preliminary Structural Analysis}},
  booktitle = {Proceedings of the Student Research Workshop at the 14th Conference of the European Chapter of the Association for Computational Linguistics},
  month     = {April},
  year      = {2014},
  address   = {Gothenburg, Sweden},
  publisher = {Association for Computational Linguistics},
  pages     = {96--105},
  url       = {http://www.aclweb.org/anthology/E14-3011}
}
"""

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'book_corpus.txt')

_URL_FILE_STATS = dict()
for line in open(_URL_FILE_STATS_PATH, 'r', encoding='utf-8'):
    url, hex_hash, file_size = line.strip().split()
    _URL_FILE_STATS[url] = hex_hash


# The Gutenberg dataset is downloaded from:
# https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html, and
# is a small subset of the Project Gutenberg corpus
# The original link for
# downloading is https://drive.google.com/file/d/0B2Mzhc7popBga2RkcWZNcjlRTGM/edit?usp=sharing

_URLS = {
    'gutenberg':
        'https://gluonnlp-numpy-data.s3-us-west-2.amazonaws.com/pretrain_corpus/Gutenberg.zip',
}


def get_parser():
    parser = argparse.ArgumentParser(description='Download and Prepare the BookCorpus dataset.')
    parser.add_argument('--dataset', type=str, choices=['gutenberg'], default='gutenberg')
    parser.add_argument('--mode', type=str, default='raw', choices=['raw', 'format'],
                        help='Specify the mode for preparing the data.'
                             ' "raw" means to download and extract the books into the output'
                             ' folder, each file is a book and the filename is the tile of the '
                             'book. "format" means to format the extracted txt files for '
                             'usage of pretraining.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='The directory to save the dataset. Default is the same as the'
                             ' dataset.')
    parser.add_argument('--cache-path', type=str,
                        default=os.path.join(get_data_home_dir(), 'book_corpus'),
                        help='The temporary path to download the compressed dataset.')
    return parser


def main(args):
    url = _URLS[args.dataset]
    file_hash = _URL_FILE_STATS[url]
    target_download_location = os.path.join(args.cache_path,
                                            os.path.basename(url))
    download(url, target_download_location, sha1_hash=file_hash)
    save_dir = args.dataset if args.save_dir is None else args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if args.dataset == 'gutenberg':
        if args.mode == 'raw':
            with zipfile.ZipFile(target_download_location) as f:
                for name in f.namelist():
                    if name.endswith('.txt'):
                        filename = os.path.basename(name)
                    f.extract(name, os.path.join(save_dir, filename))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
