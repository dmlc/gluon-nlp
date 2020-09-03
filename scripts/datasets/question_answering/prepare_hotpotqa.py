import os
import argparse
from gluonnlp.utils.misc import download, load_checksum_stats
from gluonnlp.base import get_data_home_dir

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_BASE_DATASET_PATH = os.path.join(get_data_home_dir(), 'hotpotqa')
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'hotpotqa.txt')
_URL_FILE_STATS = load_checksum_stats(_URL_FILE_STATS_PATH)


_CITATIONS = """
@inproceedings{yang2018hotpotqa,
  title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle={Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year={2018}
}

"""

_URLS = {
    'train': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json',
    'dev_fullwiki': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json',
    'dev_distractor': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
    'test_fullwiki': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json',
}


def get_parser():
    parser = argparse.ArgumentParser(description='Downloading the HotpotQA Dataset.')
    parser.add_argument('--save-path', type=str, default='hotpotqa')
    parser.add_argument('--cache-path', type=str, default=_BASE_DATASET_PATH,
                        help='The path to download the dataset.')
    parser.add_argument('--overwrite', action='store_true')
    return parser


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
