import statistics
import nltk
import multiprocessing
from collections import defaultdict
from itertools import islice
import time
import os



def segment_sentences(input):
    articles, output_names, rank, segmenter = input
    print('process {} needs to segment {} articles'.format(rank, len(articles)))
    sentences = {}
    for i, article in enumerate(articles):
        sentences[article] = segmenter.segment_string(articles[article])

        if i % 100 == 0:
            print('process {} finish article {}'.format(rank, i))

    print('process {} finish segment'.format(rank))
    # return_dict.update(sentences)
    total_length = 0
    for article_id in sentences:
        total_length += len(sentences[article_id])

    # try to average size of output size
    ideal_length = total_length // len(output_names) + 1
    output_list = defaultdict(lambda: [])
    output_index = 0
    current_length = 0

    for article_id in sentences:
        current_length += len(sentences[article_id])
        output_list[output_names[output_index]].append(article_id)
        if current_length >= ideal_length:
            output_index += 1
            current_length = 0

    print('process {} start to write to disk'.format(rank))
    for output_name in output_names:
        with open(output_name, mode='w', newline='\n') as f:
            for article_id in output_list[output_name]:
                for line in sentences[article_id]:
                    f.write(line + '\n')

                f.write('\n')

    print('process {} finish to write to disk'.format(rank))

class NLTKSegmenter:
    def __init(self):
        pass

    def segment_string(self, article):
        return nltk.tokenize.sent_tokenize(article)




class Sharding:
    def __init__(self, input_files, output_name_prefix, n_training_shards, n_test_shards, fraction_test_set, segmenting_num_worker):
        assert len(input_files) > 0, 'The input file list must contain at least one file.'
        assert n_training_shards > 0, 'There must be at least one output shard.'
        assert n_test_shards > 0, 'There must be at least one output shard.'

        self.n_training_shards = n_training_shards
        self.n_test_shards = n_test_shards
        self.fraction_test_set = fraction_test_set
        self.segmenting_num_worker = segmenting_num_worker

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


    def segment_articles_into_sentences(self):
        print('Start: Sentence Segmentation')
        segmenter = NLTKSegmenter()
        if len(self.articles) is 0:
            self.load_articles()

        assert len(self.articles) is not 0, 'Please check that input files are present and contain data.'



        def chunks(data, names, size=len(self.articles), name_size = 1):
            it = iter(data)
            it_name = iter(names)
            for i in range(0, len(data), size):
                yield ({k: data[k] for k in islice(it, size)}, [p for p in islice(it_name, name_size)])



        n_processes = self.segmenting_num_worker  # in addition to the main process, total = n_proc+1
        pool = multiprocessing.Pool(n_processes)
        rank = 0
        args=[]
        for item, name_item in chunks(self.articles, self.output_training_files,
                                      (len(self.articles) - 1) // (n_processes) + 1,
                                      (len(self.output_training_files) - 1) // (n_processes) + 1):
            args.append((item, name_item, rank, segmenter))
            rank += 1
        pool.map(segment_sentences, args)


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


nltk.download('punkt')

