"""Prepare the bookcorpus dataset that contains raw text file."""

import os
import time
import tarfile
import argparse
import glob
from gluonnlp.utils.misc import download, load_checksum_stats
from collections import defaultdict
from itertools import islice
import time

import multiprocessing
import statistics
import nltk

nltk.download('punkt')

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'bookcorpus.txt')
_URL_FILE_STATS = load_checksum_stats(_URL_FILE_STATS_PATH)
_URLS = {
    'books1':
        'https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz',
}

class NLTKSegmenter:
    def __init(self):
        pass

    def segment_string(self, article):
        return nltk.tokenize.sent_tokenize(article)


class Sharding:
    def __init__(self, input_files, output_name_prefix, n_training_shards, n_test_shards, fraction_test_set):
        assert len(input_files) > 0, 'The input file list must contain at least one file.'
        assert n_training_shards > 0, 'There must be at least one output shard.'
        assert n_test_shards > 0, 'There must be at least one output shard.'

        self.n_training_shards = n_training_shards
        self.n_test_shards = n_test_shards
        self.fraction_test_set = fraction_test_set

        self.input_files = input_files

        self.output_name_prefix = output_name_prefix
        self.output_training_identifier = '_training'
        self.output_test_identifier = '_test'
        self.output_file_extension = '.txt'

        self.articles = {}    # key: integer identifier, value: list of articles
        self.sentences = {}    # key: integer identifier, value: list of sentences
        self.output_training_files = {}    # key: filename, value: list of articles to go into file
        self.output_test_files = {}  # key: filename, value: list of articles to go into file

        self.init_output_files()


    # Remember, the input files contain one article per line (the whitespace check is to skip extraneous blank lines)
    def load_articles(self):
        print('Start: Loading Articles')

        global_article_count = 0
        for input_file in self.input_files:
            print('input file:', input_file)
            with open(input_file, mode='r', newline='\n') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        self.articles[global_article_count] = line.rstrip()
                        global_article_count += 1

        print('End: Loading Articles: There are', len(self.articles), 'articles.')


    def segment_articles_into_sentences(self, segmenter):
        print('Start: Sentence Segmentation')
        if len(self.articles) is 0:
            self.load_articles()

        assert len(self.articles) is not 0, 'Please check that input files are present and contain data.'

        # TODO: WIP: multiprocessing (create independent ranges and spawn processes)
        use_multiprocessing = 'manager'

        def chunks(data, size=len(self.articles)):
            it = iter(data)
            for i in range(0, len(data), size):
                yield {k: data[k] for k in islice(it, size)}

        if use_multiprocessing == 'manager':
            pass
            ##To do: debug
            time1 = time.time()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            jobs = []
            n_processes = 16    # in addition to the main process, total = n_proc+1

            def work(articles, return_dict, rank):
                print(len(articles))
                sentences = {}
                for i, article in enumerate(articles):
                    sentences[article] = segmenter.segment_string(articles[article])

                    if i % 100 == 0:
                        print('Segmenting article', i, rank)

                print('finish:', rank)
                return_dict.update(sentences)
            rank = 0
            for item in chunks(self.articles, len(self.articles)//(n_processes)+1):
                p = multiprocessing.Process(target=work, args=(item, return_dict,rank))
                rank+=1
                # Busy wait
                while len(jobs) >= n_processes:
                    pass

                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()
            self.sentences.update(return_dict)

            time2 = time.time()

            print('finish spend:', time2-time1)
        elif use_multiprocessing == 'queue':
            work_queue = multiprocessing.Queue()
            jobs = []

            for item in chunks(self.articles, len(self.articles)):
                pass

        else:
            # serial option
            time1= time.time()
            for i, article in enumerate(self.articles):
                self.sentences[i] = segmenter.segment_string(self.articles[article])

                if i % 100 == 0:
                    print('Segmenting article', i)
            time2 = time.time()
            print('segmentation spend:', time2-time1)
        print('End: Sentence Segmentation')


    def init_output_files(self):
        print('Start: Init Output Files')
        assert len(self.output_training_files) is 0, 'Internal storage self.output_files already contains data. This function is intended to be used by the constructor only.'
        assert len(self.output_test_files) is 0, 'Internal storage self.output_files already contains data. This function is intended to be used by the constructor only.'

        for i in range(self.n_training_shards):
            name = self.output_name_prefix + self.output_training_identifier + '_' + str(i) + self.output_file_extension
            self.output_training_files[name] = []

        for i in range(self.n_test_shards):
            name = self.output_name_prefix + self.output_test_identifier + '_' + str(i) + self.output_file_extension
            self.output_test_files[name] = []

        print('End: Init Output Files')


    def get_sentences_per_shard(self, shard):
        result = 0
        for article_id in shard:
            result += len(self.sentences[article_id])

        return result


    def distribute_articles_over_shards(self):
        print('Start: Distribute Articles Over Shards')
        assert len(self.articles) >= self.n_training_shards + self.n_test_shards, 'There are fewer articles than shards. Please add more data or reduce the number of shards requested.'

        # Create dictionary with - key: sentence count per article, value: article id number
        sentence_counts = defaultdict(lambda: [])

        max_sentences = 0
        total_sentences = 0
        #print(self.sentences.keys())

        for article_id in self.sentences:
            current_length = len(self.sentences[article_id])
            sentence_counts[current_length].append(article_id)
            #if current_length>max_sentences:
            #    print(article_id, current_length)
            max_sentences = max(max_sentences, current_length)
            total_sentences += current_length
        #print(max_sentences, sentence_counts, sentence_counts[max_sentences])
        n_sentences_assigned_to_training = int((1 - self.fraction_test_set) * total_sentences)
        nominal_sentences_per_training_shard = n_sentences_assigned_to_training // self.n_training_shards
        nominal_sentences_per_test_shard = (total_sentences - n_sentences_assigned_to_training) // self.n_test_shards

        consumed_article_set = set({})
        unused_article_set = set(self.articles.keys())

        # Make first pass and add one article worth of lines per file
        for file in self.output_training_files:
            current_article_id = sentence_counts[max_sentences][-1]
            sentence_counts[max_sentences].pop(-1)
            self.output_training_files[file].append(current_article_id)
            consumed_article_set.add(current_article_id)
            unused_article_set.remove(current_article_id)

            # Maintain the max sentence count
            while len(sentence_counts[max_sentences]) == 0 and max_sentences > 0:
                max_sentences -= 1

            if len(self.sentences[current_article_id]) > nominal_sentences_per_training_shard:
                nominal_sentences_per_training_shard = len(self.sentences[current_article_id])
                print('Warning: A single article contains more than the nominal number of sentences per training shard.')

        for file in self.output_test_files:
            current_article_id = sentence_counts[max_sentences][-1]
            sentence_counts[max_sentences].pop(-1)
            self.output_test_files[file].append(current_article_id)
            consumed_article_set.add(current_article_id)
            unused_article_set.remove(current_article_id)

            # Maintain the max sentence count
            while len(sentence_counts[max_sentences]) == 0 and max_sentences > 0:
                max_sentences -= 1

            if len(self.sentences[current_article_id]) > nominal_sentences_per_test_shard:
                nominal_sentences_per_test_shard = len(self.sentences[current_article_id])
                print('Warning: A single article contains more than the nominal number of sentences per test shard.')

        training_counts = []
        test_counts = []

        for shard in self.output_training_files:
            training_counts.append(self.get_sentences_per_shard(self.output_training_files[shard]))

        for shard in self.output_test_files:
            test_counts.append(self.get_sentences_per_shard(self.output_test_files[shard]))

        training_median = statistics.median(training_counts)
        test_median = statistics.median(test_counts)

        # Make subsequent passes over files to find articles to add without going over limit
        history_remaining = []
        n_history_remaining = 4

        while len(consumed_article_set) < len(self.articles):
            for fidx, file in enumerate(self.output_training_files):
                nominal_next_article_size = min(nominal_sentences_per_training_shard - training_counts[fidx], max_sentences)

                # Maintain the max sentence count
                while len(sentence_counts[max_sentences]) == 0 and max_sentences > 0:
                    max_sentences -= 1

                while len(sentence_counts[nominal_next_article_size]) == 0 and nominal_next_article_size > 0:
                    nominal_next_article_size -= 1

                if nominal_next_article_size not in sentence_counts or nominal_next_article_size is 0 or training_counts[fidx] > training_median:
                    continue    # skip adding to this file, will come back later if no file can accept unused articles

                current_article_id = sentence_counts[nominal_next_article_size][-1]
                sentence_counts[nominal_next_article_size].pop(-1)

                self.output_training_files[file].append(current_article_id)
                consumed_article_set.add(current_article_id)
                unused_article_set.remove(current_article_id)

            for fidx, file in enumerate(self.output_test_files):
                nominal_next_article_size = min(nominal_sentences_per_test_shard - test_counts[fidx], max_sentences)

                # Maintain the max sentence count
                while len(sentence_counts[max_sentences]) == 0 and max_sentences > 0:
                    max_sentences -= 1

                while len(sentence_counts[nominal_next_article_size]) == 0 and nominal_next_article_size > 0:
                    nominal_next_article_size -= 1

                if nominal_next_article_size not in sentence_counts or nominal_next_article_size is 0 or test_counts[fidx] > test_median:
                    continue    # skip adding to this file, will come back later if no file can accept unused articles

                current_article_id = sentence_counts[nominal_next_article_size][-1]
                sentence_counts[nominal_next_article_size].pop(-1)

                self.output_test_files[file].append(current_article_id)
                consumed_article_set.add(current_article_id)
                unused_article_set.remove(current_article_id)

            # If unable to place articles a few times, bump up nominal sizes by fraction until articles get placed
            if len(history_remaining) == n_history_remaining:
                history_remaining.pop(0)
            history_remaining.append(len(unused_article_set))

            history_same = True
            for i in range(1, len(history_remaining)):
                history_same = history_same and (history_remaining[i-1] == history_remaining[i])

            if history_same:
                nominal_sentences_per_training_shard += 1
                # nominal_sentences_per_test_shard += 1

            training_counts = []
            test_counts = []
            for shard in self.output_training_files:
                training_counts.append(self.get_sentences_per_shard(self.output_training_files[shard]))

            for shard in self.output_test_files:
                test_counts.append(self.get_sentences_per_shard(self.output_test_files[shard]))

            training_median = statistics.median(training_counts)
            test_median = statistics.median(test_counts)

            print('Distributing data over shards:', len(unused_article_set), 'articles remaining.')


        if len(unused_article_set) != 0:
            print('Warning: Some articles did not make it into output files.')


        for shard in self.output_training_files:
            print('Training shard:', self.get_sentences_per_shard(self.output_training_files[shard]))

        for shard in self.output_test_files:
            print('Test shard:', self.get_sentences_per_shard(self.output_test_files[shard]))

        print('End: Distribute Articles Over Shards')


    def write_shards_to_disk(self):
        print('Start: Write Shards to Disk')
        for shard in self.output_training_files:
            self.write_single_shard(shard, self.output_training_files[shard])

        for shard in self.output_test_files:
            self.write_single_shard(shard, self.output_test_files[shard])

        print('End: Write Shards to Disk')


    def write_single_shard(self, shard_name, shard):
        with open(shard_name, mode='w', newline='\n') as f:
            for article_id in shard:
                for line in self.sentences[article_id]:
                    f.write(line + '\n')

                f.write('\n')  # Line break between articles

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
    parser.add_argument("--segment_sentences", action='store_true',
                        help="directory for downloaded  files")
    return parser



def main(args):
    '''
    url =_URLS['books1']
    file_hash = _URL_FILE_STATS[url]
    target_download_location = os.path.join(args.output,
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
    format = BookscorpusTextFormatting('BookCorpus/books1/epubtxt', 'BookCorpus/bookcorpus.txt')
    format.merge()
    print("end format")
    '''
    if args.segment_sentences:
        print("start to transfer bookcorpus to one sentence per line")
        t1 = time.time()
        segmenter = NLTKSegmenter()
        if not os.path.exists('./BookCorpus/one_sentence_per_line/'):
            os.mkdir('./BookCorpus/one_sentence_per_line/')
        sharding = Sharding(['BookCorpus/bookcorpus.txt'], 'BookCorpus/one_sentence_per_line/', 128, 1, 0)

        sharding.load_articles()
        sharding.segment_articles_into_sentences(segmenter)
        sharding.distribute_articles_over_shards()
        sharding.write_shards_to_disk()
        t2 = time.time()
        print("transfer cost:{}".format(t2-t1))




def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

