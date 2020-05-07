import argparse
import os
import tarfile
from gluonnlp.base import get_data_home_dir
from gluonnlp.utils.misc import download
import zipfile

_CITATIONS = """
@phdthesis{raffel2016learning,
  title={Learning-based methods for comparing sequences, with applications to audio-to-midi alignment and matching},
  author={Raffel, Colin},
  year={2016},
  school={Columbia University}
}

@inproceedings{hawthorne2018enabling,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7},
}
"""


_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_BASE_DATASET_PATH = os.path.join(get_data_home_dir(), 'music_midi_data')

_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'music_midi.txt')
_URL_FILE_STATS = dict()
for line in open(_URL_FILE_STATS_PATH, 'r', encoding='utf-8'):
    url, hex_hash, file_size = line.strip().split()
    _URL_FILE_STATS[url] = hex_hash


_URLS = {
    'lmd_full': 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz',
    'lmd_matched': 'http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz',
    'lmd_aligned': 'http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz',
    'clean_midi': 'http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz',
    'maestro_v1': 'https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip',
    'maestro_v2': 'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
    'geocities': 'https://archive.org/download/archiveteam-geocities-midi-collection-2009/2009.GeoCities.MIDI.ArchiveTeam.zip'
}


def get_parser():
    parser = argparse.ArgumentParser(description='Download the Music Midi Datasets.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['lmd_full', 'lmd_matched', 'lmd_aligned', 'clean_midi',
                                 'maestro_v1', 'maestro_v2', 'geocities'],
                        help='The dataset to download.')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='The directory to save the dataset.'
                             ' By default, it will save to a folder with the same name as the '
                             'dataset')
    parser.add_argument('--overwrite', action='store_true',
                        help='Whether to overwrite the directory.')
    parser.add_argument('--cache-path', type=str, default=_BASE_DATASET_PATH,
                        help='The temporary path to download the compressed dataset.')
    return parser


def main(args):
    # Download the data
    url = _URLS[args.dataset]
    file_hash = _URL_FILE_STATS[url]
    target_download_location = os.path.join(args.cache_path, os.path.basename(url))
    download(url, target_download_location, sha1_hash=file_hash)
    if args.save_dir is None:
        save_dir = args.dataset
    else:
        save_dir = args.save_dir
    if not args.overwrite and os.path.exists(save_dir):
        print('{} found, skip! Turn on --overwrite to force overwrite'.format(save_dir))
    if args.dataset == 'lmd_full':
        with tarfile.open(target_download_location) as f:
            f.extractall(save_dir)
    elif args.dataset == 'lmd_matched':
        with tarfile.open(target_download_location) as f:
            f.extractall(save_dir)
    elif args.dataset == 'lmd_aligned':
        with tarfile.open(target_download_location) as f:
            f.extractall(save_dir)
    elif args.dataset == 'clean_midi':
        with tarfile.open(target_download_location) as f:
            f.extractall(save_dir)
    elif args.dataset == 'maestro_v1':
        with zipfile.ZipFile(target_download_location, 'r') as fobj:
            fobj.extractall(save_dir)
    elif args.dataset == 'maestro_v2':
        with zipfile.ZipFile(target_download_location, 'r') as fobj:
            fobj.extractall(save_dir)
    elif args.dataset == 'geocities':
        with zipfile.ZipFile(target_download_location, 'r') as fobj:
            fobj.extractall(save_dir)
    else:
        raise NotImplementedError


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
