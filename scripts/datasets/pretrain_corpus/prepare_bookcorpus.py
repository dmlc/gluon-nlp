"""Prepare the bookcorpus dataset that contains raw text file."""

import os
import time
import tarfile
import argparse
from gluonnlp.utils.misc import download, load_checksum_stats

_DOWNLOAD_URL\
    = "https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz"

#_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
#_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'bookcorpus.txt')
#_URL_FILE_STATS = load_checksum_stats(_URL_FILE_STATS_PATH)
#print(_URL_FILE_STATS)

#exit()


def get_parser():
    parser = argparse.ArgumentParser(description='Download the raw txt BookCorpus')
    parser.add_argument("-o", "--output", default="BookCorpus",
                        help="directory for downloaded  files")
    return parser



def main(args):
    url =_DOWNLOAD_URL
    target_download_location = os.path.join(args.output,
                                            os.path.basename(url))
    download(url, target_download_location)
    tar = tarfile.open(target_download_location)
    names = tar.getnames()
    print('Start unarchiving raw text files')
    start_time = time.time()
    for name in names:
        tar.extract(name, path=args.output)
    tar.close()
    print("Done unarchiving within {:.2f} seconds".format(time.time() - start_time))




def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

