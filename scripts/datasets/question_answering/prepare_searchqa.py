import os
import argparse
from gluonnlp.registry import DATA_PARSER_REGISTRY, DATA_MAIN_REGISTRY
from gluonnlp.utils.misc import download, load_checksum_stats
from gluonnlp.base import get_data_home_dir

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_BASE_DATASET_PATH = os.path.join(get_data_home_dir(), 'searchqa')
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'searchqa.txt')
_URL_FILE_STATS = load_checksum_stats(_URL_FILE_STATS_PATH)


_CITATIONS = """
@article{dunn2017searchqa,
  title={Searchqa: A new q\&a dataset augmented with context from a search engine},
  author={Dunn, Matthew and Sagun, Levent and Higgins, Mike and Guney, V Ugur and Cirik, Volkan and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:1704.05179},
  year={2017}
}

"""

_URLS = {
    'train': 's3://gluonnlp-numpy-data/datasets/question_answering/searchqa/train.txt',
    'val': 's3://gluonnlp-numpy-data/datasets/question_answering/searchqa/val.txt',
    'test': 's3://gluonnlp-numpy-data/datasets/question_answering/searchqa/test.txt'
}


@DATA_PARSER_REGISTRY.register('prepare_searchqa')
def get_parser():
    parser = argparse.ArgumentParser(description='Downloading the SearchQA Dataset.')
    parser.add_argument('--save-path', type=str, default='searchqa')
    parser.add_argument('--cache-path', type=str, default=_BASE_DATASET_PATH,
                        help='The path to download the dataset.')
    parser.add_argument('--overwrite', action='store_true')
    return parser


@DATA_MAIN_REGISTRY.register('prepare_searchqa')
def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    for url in _URLS.values():
        file_name = url[url.rfind('/') + 1:]
        file_hash = _URL_FILE_STATS[url]
        download(url, path=os.path.join(args.cache_path, file_name), sha1_hash=file_hash)
        if not os.path.exists(os.path.join(args.save_path, file_name))\
                or (args.overwrite and args.save_path != args.cache_path):
            os.symlink(os.path.join(args.cache_path, file_name),
                       os.path.join(args.save_path, file_name))


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
