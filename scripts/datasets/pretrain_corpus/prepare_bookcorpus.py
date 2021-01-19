"""Prepare the bookcorpus dataset that contains raw text file."""

import os
import tarfile
import argparse
import glob
from gluonnlp.base import get_data_home_dir
from gluonnlp.utils.misc import download, load_checksum_stats
from collections import defaultdict
from itertools import islice
import time
import multiprocessing
import statistics
import nltk
import sys

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_TARGET_PATH = os.path.join(_CURR_DIR, '../../processing/')
sys.path.append(_TARGET_PATH)
from segment_sentences import Sharding, segment_sentences, NLTKSegmenter



_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'bookcorpus.txt')
_URL_FILE_STATS = load_checksum_stats(_URL_FILE_STATS_PATH)
_URLS = {
    'books1':
        'https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz',
}


class BookscorpusTextFormatting:
    def __init__(self, books_path, output_filename, recursive = False):
        self.books_path = books_path
        self.recursive = recursive
        self.output_filename = output_filename


    # This puts one book per line
    def merge(self):
        with open(self.output_filename, mode='w', newline='\n') as ofile:
            for filename in glob.glob(self.books_path + '/' + '*.txt', recursive=True):
                with open(filename, mode='r', encoding='utf-8-sig', newline='\n') as file:
                    for line in file:
                        if line.strip() != '':
                            ofile.write(line.strip() + ' ')
                ofile.write("\n\n")



def get_parser():
    parser = argparse.ArgumentParser(description='Download the raw txt BookCorpus')
    parser.add_argument("-o", "--output", default="BookCorpus",
                        help="directory for downloaded  files")
    parser.add_argument("--segment_num_worker", type=int, default=8,
                        help="process num when segmenting articles")
    parser.add_argument("--segment_sentences", action='store_true',
                        help="directory for downloaded  files")
    parser.add_argument('--cache-path', type=str,
                        default=os.path.join(get_data_home_dir(), 'bookcorpus'),
                        help='The temporary path to download the dataset.')
    return parser



def main(args):
    url =_URLS['books1']
    file_hash = _URL_FILE_STATS[url]
    target_download_location = os.path.join(args.cache_path,
                                            os.path.basename(url))
    download(url, target_download_location, sha1_hash=file_hash)
    tar = tarfile.open(target_download_location)
    names = tar.getnames()
    print('Start unarchiving raw text files')
    start_time = time.time()
    for name in names:
        tar.extract(name, path=args.output)
    tar.close()
    print("Done unarchiving within {:.2f} seconds".format(time.time() - start_time))
    print("start transfer to one article per line")
    input_name = os.path.join(args.output, 'books1/epubtxt/')
    output_name = os.path.join(args.output,'bookcorpus.txt' )
    format = BookscorpusTextFormatting(input_name, output_name)
    format.merge()
    print("end format")
    if args.segment_sentences:
        print("start to transfer bookcorpus to one sentence per line")
        t1 = time.time()

        input_name = os.path.join(args.output, 'bookcorpus.txt')
        output_name = os.path.join(args.output, 'one_sentence_per_line/')
        if not os.path.exists(output_name):
            os.mkdir(output_name)
        sharding = Sharding([input_name], output_name, 128, 1, 0 ,args.segment_num_worker)

        sharding.load_articles()
        sharding.segment_articles_into_sentences()
        t2 = time.time()
        print("transfer cost:{}".format(t2-t1))




def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

