import os
import argparse
from gluonnlp.utils.misc import download, load_checksum_stats
from gluonnlp.base import get_data_home_dir

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_BASE_DATASET_PATH = os.path.join(get_data_home_dir(), 'squad')
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'squad.txt')
_URL_FILE_STATS = load_checksum_stats(_URL_FILE_STATS_PATH)


_CITATIONS = """
@inproceedings{rajpurkar2016squad,
  title={Squad: 100,000+ questions for machine comprehension of text},
  author={Rajpurkar, Pranav and Zhang, Jian and Lopyrev, Konstantin and Liang, Percy},
  booktitle={EMNLP},
  year={2016}
}

@inproceedings{rajpurkar2018know,
  title={Know What You Don't Know: Unanswerable Questions for SQuAD},
  author={Rajpurkar, Pranav and Jia, Robin and Liang, Percy},
  booktitle={ACL},
  year={2018}
}

"""

_URLS = {
    '1.1': {
        'train': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
        'dev': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
    },
    '2.0': {
        'train': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
        'dev': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'
    }
}


def get_parser():
    parser = argparse.ArgumentParser(description='Downloading the SQuAD Dataset.')
    parser.add_argument('--version', type=str, choices=['1.1', '2.0'], default='1.1',
                        help='Version of the squad dataset.')
    parser.add_argument('--save-path', type=str, default='squad')
    parser.add_argument('--cache-path', type=str, default=_BASE_DATASET_PATH,
                        help='The path to download the dataset.')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main(args):
    train_url = _URLS[args.version]['train']
    dev_url = _URLS[args.version]['dev']
    train_file_name = train_url[train_url.rfind('/') + 1:]
    dev_file_name = dev_url[dev_url.rfind('/') + 1:]
    download(train_url, path=os.path.join(args.cache_path, train_file_name))
    download(dev_url, path=os.path.join(args.cache_path, dev_file_name))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, train_file_name))\
            or (args.overwrite and args.save_path != args.cache_path):
        os.symlink(os.path.join(args.cache_path, train_file_name),
                   os.path.join(args.save_path, train_file_name))
    if not os.path.exists(os.path.join(args.save_path, dev_file_name))\
            or (args.overwrite and args.save_path != args.cache_path):
        os.symlink(os.path.join(args.cache_path, dev_file_name),
                   os.path.join(args.save_path, dev_file_name))


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
