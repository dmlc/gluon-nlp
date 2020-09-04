import argparse
import os
import multiprocessing
import time
import numpy as np
import warnings
import re
from gluonnlp.data.filtering import MosesNormalizer
from gluonnlp.data.tokenizers import MosesTokenizer, BaseTokenizer,\
                                     WhitespaceTokenizer, JiebaTokenizer
from typing import List, Union, Optional
re._MAXCACHE = 1024


def get_tokenizer(tokenizer, lang=None):
    if isinstance(tokenizer, BaseTokenizer):
        return tokenizer
    else:
        if tokenizer == 'moses':
            return MosesTokenizer(lang=lang)
        elif tokenizer == 'whitespace':
            return WhitespaceTokenizer()
        elif tokenizer == 'jieba':
            return JiebaTokenizer()
        else:
            raise NotImplementedError


def check_both_latin1(src_sentence: str, tgt_sentence: str) -> bool:
    """Check whether the sentence pair can all be encoded in latin1

    This is used in
    https://github.com/mlperf/training/blob/master/rnn_translator/pytorch/scripts/filter_dataset.py

    The idea is to filter the sentences with rare unicode glyphs and are unlikely to be en-de

    Returns
    -------
    ret
        Whether both sentences are latin1
    """
    try:
        src_sentence.encode('latin1')
        tgt_sentence.encode('latin1')
    except UnicodeEncodeError:
        return False
    else:
        return True


def get_line_byte_start(corpus_path: str) -> np.ndarray:
    """Get the start position of each lines in terms of bytes so that we can use seek + read to
     load an arbitrary line.

    Parameters
    ----------
    corpus_path
        The path of the corpus

    Returns
    -------
    line_pos
        Shape (#Lens + 1,)
    """
    line_pos = [0]
    with open(corpus_path, 'rb') as in_f:
        pos = 0
        for line in in_f:
            pos += len(line)
            line_pos.append(pos)
    return np.array(line_pos, dtype=np.int64)


class ParallelCorpusProcessor:
    """Process a pair of corpus.

    This largely recovers the functionality of 'clean-corpus-n.perl' in mosesdecoder.
    The difference is that it is customizable with pure python.

    By default, we will perform the following pre-processing pipeline.
    Each stage could be turned on/off and specialized based on the input arguments.
    Also, you may directly revise the code and write your own processing script.

    1. Normalize sentence
    2. Pre-filter
    3. Tokenize the sentence
    4. Filter the sentence based on different rules
        3.1 Remove pairs where `max(len(lhs) / len(rhs), len(rhs) / len(lhs) > max_ratio`
        3.2 Remove pairs where not `min_max_words <= len(lhs) <= max_num_words` and
                                   `min_max_words <= len(rhs) <= max_num_words`
    """
    def __init__(self, src_lang: str, tgt_lang: str,
                 normalize: bool = True,
                 src_tokenizer: Union[str, BaseTokenizer] = 'whitespace',
                 tgt_tokenizer: Union[str, BaseTokenizer] = 'whitespace',
                 max_ratio: Optional[float] = None,
                 min_num_words: Optional[int] = None,
                 max_num_words: Optional[int] = None,
                 discard_non_latin1: bool = False):
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        if normalize:
            self._src_normalizer = MosesNormalizer(lang=src_lang)
            self._tgt_normalizer = MosesNormalizer(lang=tgt_lang)
        self._src_tokenizer = get_tokenizer(src_tokenizer, src_lang)
        self._tgt_tokenizer = get_tokenizer(tgt_tokenizer, tgt_lang)
        self._max_ratio = max_ratio
        self._min_num_words = min_num_words
        self._max_num_words = max_num_words
        self._discard_non_latin1 = discard_non_latin1

    def process_chunk(self, args):
        src_path, src_chunk_start, src_chunk_size, tgt_path, tgt_chunk_start, tgt_chunk_size = args
        processed_src_lines = []
        processed_tgt_lines = []
        with open(src_path, 'rb') as src_in_f:
            with open(tgt_path, 'rb') as tgt_in_f:
                # Read chunk from source and target
                src_in_f.seek(src_chunk_start)
                src_lines = src_in_f.read(src_chunk_size)
                tgt_in_f.seek(tgt_chunk_start)
                tgt_lines = tgt_in_f.read(tgt_chunk_size)
                src_lines = src_lines.splitlines()
                tgt_lines = tgt_lines.splitlines()
                unfiltered_line_num = len(src_lines)
                for src_line, tgt_line in zip(src_lines, tgt_lines):
                    src_line = src_line.decode('utf-8').strip()
                    tgt_line = tgt_line.decode('utf-8').strip()
                    # 1. Normalize
                    src_line = self._src_normalizer(src_line)
                    tgt_line = self._tgt_normalizer(tgt_line)
                    # 2. Filter after normalization.
                    if self._discard_non_latin1:
                        if not check_both_latin1(src_line, tgt_line):
                            continue
                    # 3. Tokenize the sentence
                    src_tokens = self._src_tokenizer.encode(src_line)
                    tgt_tokens = self._tgt_tokenizer.encode(tgt_line)
                    # 4. Filter after tokenization. Filter with multiple rules
                    if len(src_tokens) == 0 or len(tgt_tokens) == 0:
                        continue
                    if self._max_ratio is not None:
                        if max(len(src_tokens) / len(tgt_tokens),
                               len(tgt_tokens) / len(src_tokens)) > self._max_ratio:
                            continue
                    if self._max_num_words is not None:
                        if len(src_tokens) > self._max_num_words or\
                                len(tgt_tokens) > self._max_num_words:
                            continue
                    if self._min_num_words is not None:
                        if len(src_tokens) < self._min_num_words\
                                or len(tgt_tokens) < self._min_num_words:
                            continue
                    processed_src_lines.append(' '.join(src_tokens))
                    processed_tgt_lines.append(' '.join(tgt_tokens))
        return processed_src_lines, processed_tgt_lines, unfiltered_line_num

    def process_parallel_corpus(self, src_corpus_paths: List[str],
                                tgt_corpus_paths: List[str],
                                src_out_path: str, tgt_out_path: str,
                                chunk_size: int = 1024 * 1024,
                                num_process: int = 8) -> int:
        """Preprocess the parallel corpus

        Parameters
        ----------
        src_corpus_paths
            Source corpus paths
        tgt_corpus_paths
            Target corpus paths
        src_out_path
            Write the results to the source output path
        tgt_out_path
            Write the results to the target output path
        chunk_size
            Approximately split the corpus files into multiple chunks
        num_process
            The number of process

        Returns
        -------
        line_count
            The number of lines in the final filtered file
        """
        start = time.time()
        total_line_count = 0
        filtered_line_count = 0

        def chunk_iterator(step=10):
            for src_path, tgt_path in zip(src_corpus_paths, tgt_corpus_paths):
                src_line_pos = get_line_byte_start(src_path)
                tgt_line_pos = get_line_byte_start(tgt_path)
                src_line_size = src_line_pos[1:] - src_line_pos[:-1]
                tgt_line_size = tgt_line_pos[1:] - tgt_line_pos[:-1]
                num_src_lines = src_line_pos.shape[0] - 1
                num_tgt_lines = tgt_line_pos.shape[0] - 1
                assert num_src_lines == num_tgt_lines
                src_budget = chunk_size
                tgt_budget = chunk_size
                src_chunk_start = 0
                tgt_chunk_start = 0
                src_chunk_size = 0
                tgt_chunk_size = 0
                for i in range(0, num_src_lines, step):
                    line_batch_num = min(num_src_lines - i, step)
                    src_batch_line_size = src_line_size[i:(i + line_batch_num)].sum()
                    tgt_batch_line_size = tgt_line_size[i:(i + line_batch_num)].sum()
                    src_budget -= src_batch_line_size
                    tgt_budget -= tgt_batch_line_size
                    src_chunk_size += src_batch_line_size
                    tgt_chunk_size += tgt_batch_line_size
                    if src_budget <= 0 or tgt_budget <= 0 or i + step >= num_src_lines:
                        yield src_path, src_chunk_start, src_chunk_size,\
                              tgt_path, tgt_chunk_start, tgt_chunk_size
                        src_chunk_start += src_chunk_size
                        tgt_chunk_start += tgt_chunk_size
                        src_chunk_size = 0
                        tgt_chunk_size = 0
                        src_budget = chunk_size
                        tgt_budget = chunk_size

        with open(src_out_path, 'w', encoding='utf-8', newline='\n') as src_out_f:
            with open(tgt_out_path, 'w', encoding='utf-8', newline='\n') as tgt_out_f:
                with multiprocessing.Pool(num_process) as pool:
                    for i, (processed_src_lines, processed_tgt_lines, unfiltered_line_num) in \
                            enumerate(pool.imap(self.process_chunk, chunk_iterator())):
                        src_out_f.write('\n'.join(processed_src_lines) + '\n')
                        tgt_out_f.write('\n'.join(processed_tgt_lines) + '\n')
                        filtered_line_count += len(processed_src_lines)
                        total_line_count += unfiltered_line_num
                        if (i + 1) % 100 == 0:
                            print('Chunk {}, #Lines Processed: {}, Filtered: {}, Remain: {}'
                                  .format(i + 1, total_line_count,
                                          total_line_count - filtered_line_count,
                                          filtered_line_count))
        end = time.time()
        print('Done, #Lines {}/{}, Time spent {}'.format(filtered_line_count,
                                                         total_line_count,
                                                         end - start))
        return filtered_line_count


def get_parser():
    parser = argparse.ArgumentParser(
        description='Clean parallel corpus used in machine translation.')
    parser.add_argument('--src-corpus', type=str, nargs='+', required=True)
    parser.add_argument('--tgt-corpus', type=str, nargs='+', required=True)
    parser.add_argument('--src-lang', type=str, required=True)
    parser.add_argument('--tgt-lang', type=str, required=True)
    parser.add_argument('--src-save-path', type=str, default=None,
                        help='Path to save the cleaned and tokenized source corpus. If not set, '
                             'the default is "corpus.tok.{src_lang}"')
    parser.add_argument('--tgt-save-path', type=str, default=None,
                        help='Path to save the cleaned and tokenized source corpus. If not set, '
                             'the default is "corpus.tok.{src_lang}"')
    parser.add_argument('--src-tokenizer', type=str, default='moses')
    parser.add_argument('--tgt-tokenizer', type=str, default='moses')
    parser.add_argument('--max-ratio', type=float, default=None)
    parser.add_argument('--min-num-words', type=int, default=None)
    parser.add_argument('--max-num-words', type=int, default=None)
    parser.add_argument('--discard-non-latin1', action='store_true',
                        help='Whether to discard the sentence pair if both sentences cannot be '
                             'encoded into latin1.')
    parser.add_argument('--num-process', type=int, default=8,
                        help='number of process')
    parser.add_argument('--overwrite', action='store_true')

    return parser


def main(args):
    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    corpus_processor = ParallelCorpusProcessor(src_lang=src_lang,
                                               tgt_lang=tgt_lang,
                                               src_tokenizer=args.src_tokenizer,
                                               tgt_tokenizer=args.tgt_tokenizer,
                                               max_ratio=args.max_ratio,
                                               min_num_words=args.min_num_words,
                                               max_num_words=args.max_num_words,
                                               discard_non_latin1=args.discard_non_latin1)
    print('Clean the corpus:')
    print('   Source {}: {}'.format(src_lang, args.src_corpus))
    print('   Target {}: {}'.format(tgt_lang, args.tgt_corpus))
    if args.src_save_path is None:
        src_save_path = 'corpus.tok.{}'.format(src_lang)
    else:
        src_save_path = args.src_save_path
    if args.tgt_save_path is None:
        tgt_save_path = 'corpus.tok.{}'.format(tgt_lang)
    else:
        tgt_save_path = args.tgt_save_path
    print('Save to {} -> {} \n'
          '        {} -> {}'.format(src_lang, src_save_path, tgt_lang, tgt_save_path))
    if (os.path.exists(src_save_path) or os.path.exists(tgt_save_path)) and not args.overwrite:
        warnings.warn('{} or {} exists, skip. If you need to overwrite these two files, '
                      'rerun the script with --overwrite.'.format(src_save_path, tgt_save_path))
    else:
        corpus_processor.process_parallel_corpus(
            src_corpus_paths=args.src_corpus,
            tgt_corpus_paths=args.tgt_corpus,
            src_out_path=src_save_path,
            tgt_out_path=tgt_save_path,
            num_process=args.num_process)


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
