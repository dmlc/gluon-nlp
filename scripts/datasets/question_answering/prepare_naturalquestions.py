import logging
import os
import argparse
import ast
import gzip
from gluonnlp.utils.misc import download, load_checksum_stats
from gluonnlp.base import get_data_home_dir, get_repo_url


_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_BASE_DATASET_PATH = os.path.join(get_data_home_dir(), 'NaturalQuestions')
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'naturalquestions.txt')
_URL_FILE_STATS = load_checksum_stats(_URL_FILE_STATS_PATH)

_CITATIONS = """
@article{47761,
title	= {Natural Questions: a Benchmark for Question Answering Research},
author	= {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
year	= {2019},
journal	= {Transactions of the Association of Computational Linguistics}
}
"""

_URLS = {
    'train': get_repo_url() + 'NaturalQuestions/v1.0-simplified_simplified-nq-train.jsonl.gz',
    'dev': get_repo_url() + 'NaturalQuestions/nq-dev-all.jsonl.gz',
    # 'all': get_repo_url() + 'NaturalQuestions/*'
}




def get_parser():
    parser = argparse.ArgumentParser(description='Downloading the NaturalQuestions Dataset.')
    parser.add_argument('--all', type=ast.literal_eval, default=False)
    parser.add_argument('--save-path', type=str, default='NaturalQuestions')
    parser.add_argument('--cache-path', type=str, default=_BASE_DATASET_PATH,
                        help='The path to download the dataset.')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(args):
    def extract(gz_path):
        logging.info(f'Extracting {gz_path}, this can cost long time because the file is large')
        try:
            f_name = gz_path.replace(".gz", "")
            g_file = gzip.GzipFile(gz_path)
            open(f_name, "wb+").write(g_file.read())
            g_file.close()
            os.remove(gz_path)
        except Exception  as e:
            print(e)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.all:
        pass
    else:
        for url in _URLS.values():
            file_name = url[url.rfind('/') + 1:]
            file_hash = _URL_FILE_STATS[url]
            download(url, path=os.path.join(args.cache_path, file_name), sha1_hash=file_hash)
            if not os.path.exists(os.path.join(args.save_path, file_name))\
                    or (args.overwrite and args.save_path != args.cache_path):
                os.symlink(os.path.join(args.cache_path, file_name),
                           os.path.join(args.save_path, file_name))
            extract(os.path.join(args.save_path, file_name))


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()





