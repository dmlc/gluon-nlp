import os
import tarfile
import argparse
from gluonnlp.registry import DATA_PARSER_REGISTRY, DATA_MAIN_REGISTRY
from gluonnlp.utils.misc import download, load_checksum_stats
from gluonnlp.base import get_data_home_dir

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_BASE_DATASET_PATH = os.path.join(get_data_home_dir(), 'triviaqa')
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'triviaqa.txt')
_URL_FILE_STATS = load_checksum_stats(_URL_FILE_STATS_PATH)


_CITATIONS = """
@InProceedings{JoshiTriviaQA2017,
     author = {Joshi, Mandar and Choi, Eunsol and Weld, Daniel S. and Zettlemoyer, Luke},
     title = {TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension},
     booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
     month = {July},
     year = {2017},
     address = {Vancouver, Canada},
     publisher = {Association for Computational Linguistics},
}

"""

_URLS = {
    'rc': 'https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz',
    'unfiltered': 'https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz'
}


@DATA_PARSER_REGISTRY.register('prepare_triviaqa')
def get_parser():
    parser = argparse.ArgumentParser(description='Downloading the TriviaQA Dataset.')
    parser.add_argument('--type', type=str, choices=['rc', 'unfiltered'], default='rc',
                        help='type of the triviaqa dataset.')
    parser.add_argument('--save-path', type=str, default='triviaqa')
    parser.add_argument('--cache-path', type=str, default=_BASE_DATASET_PATH,
                        help='The path to download the dataset.')
    parser.add_argument('--overwrite', action='store_true')
    return parser


@DATA_MAIN_REGISTRY.register('prepare_triviaqa')
def main(args):

    def extract(tar_path, target_path):
        try:
            tar = tarfile.open(tar_path, "r:gz")
            file_names = tar.getnames()
            for file_name in file_names:
                tar.extract(file_name, target_path)
            tar.close()
        except Exception  as e:
            print(e)

    tar_url = _URLS[args.type]
    file_name = tar_url[tar_url.rfind('/') + 1:]
    file_hash = _URL_FILE_STATS[tar_url]
    download(tar_url, path=os.path.join(args.cache_path, file_name), sha1_hash=file_hash)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, file_name))\
            or (args.overwrite and args.save_path != args.cache_path):
        os.symlink(os.path.join(args.cache_path, file_name),
                   os.path.join(args.save_path, file_name))
    extract(os.path.join(args.save_path, file_name), args.save_path)

def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
