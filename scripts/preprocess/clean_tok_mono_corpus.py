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


# TODO(sxjscience) Consider whether to
def check_latin1(sentence: str) -> bool:
    """Check whether the sentence can be encoded in latin1

    This is used in
    https://github.com/mlperf/training/blob/master/rnn_translator/pytorch/scripts/filter_dataset.py

    The idea is to filter the sentences with rare unicode glyphs

    Returns
    -------
    ret
        Whether sentences are latin1
    """
    try:
        sentence.encode('latin1')
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


class MonoCorpusProcessor:
    """Process sentence of corpus.

    This largely recovers the functionality of 'clean-corpus-n.perl' in mosesdecoder.
    The difference is that it is customizable with pure python.

    By default, we will perform the following pre-processing pipeline.
    Each stage could be turned on/off and specialized based on the input arguments.
    Also, you may directly revise the code and write your own processing script.

    1. Normalize sentence
    2. Pre-filter
    3. Tokenize the sentence
    4. Filter the sentence based on different rules
        3.1 Remove sentences where `max(len(lhs) / len(rhs), len(rhs) / len(lhs) > max_ratio`
        3.2 Remove sentences where not `min_max_words <= len(lhs) <= max_num_words` and
                                       `min_max_words <= len(rhs) <= max_num_words`
    """
    def __init__(self, lang: str,
                 normalize: bool = True,
                 tokenizer: Union[str, BaseTokenizer] = 'whitespace',
                 min_num_words: Optional[int] = None,
                 max_num_words: Optional[int] = None,
                 discard_non_latin1: bool = False):
        self._lang = lang
        if normalize:
            self._normalizer = MosesNormalizer(lang=lang)
        self._tokenizer = get_tokenizer(tokenizer, lang)
        self._min_num_words = min_num_words
        self._max_num_words = max_num_words
        self._discard_non_latin1 = discard_non_latin1

    def process_chunk(self, args):
        path, chunk_start, chunk_size = args
        processed_lines = []
        with open(path, 'rb') as in_f:
            # Read chunk
            in_f.seek(chunk_start)
            lines = in_f.read(chunk_size)
            lines = lines.splitlines()
            unfiltered_line_num = len(lines)
            for line in lines:
                line = line.decode('utf-8').strip()
                # 1. Normalize
                line = self._normalizer(line)
                # 2. Filter after normalization.
                if self._discard_non_latin1:
                    if not check_latin1(line):
                        continue
                # 3. Tokenize the sentence
                tokens = self._tokenizer.encode(line)
                # 4. Filter after tokenization. Filter with multiple rules
                if len(tokens) == 0:
                    continue
                if self._max_num_words is not None:
                    if len(tokens) > self._max_num_words:
                        continue
                if self._min_num_words is not None:
                    if len(tokens) < self._min_num_words:
                        continue
                processed_lines.append(' '.join(tokens))
        return processed_lines, unfiltered_line_num

    def process_mono_corpus(self, 
                            corpus_paths: List[str],
                            out_path: str,
                            chunk_size: int = 1024 * 1024,
                            num_process: int = 8) -> int:
        """Preprocess the mono corpus

        Parameters
        ----------
        corpus_paths
            Corpus paths
        out_path
            Write the results to the output path
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
            for path in corpus_paths:
                line_pos = get_line_byte_start(path)
                line_size = line_pos[1:] - line_pos[:-1]
                num_lines = line_pos.shape[0] - 1
                budget = chunk_size
                chunk_start = 0
                cur_chunk_size = 0
                for i in range(0, num_lines, step):
                    line_batch_num = min(num_lines - i, step)
                    batch_line_size = line_size[i:(i + line_batch_num)].sum()
                    budget -= batch_line_size
                    cur_chunk_size += batch_line_size
                    if budget <= 0 or i + step >= num_lines:
                        yield path, chunk_start, cur_chunk_size
                        chunk_start += cur_chunk_size
                        cur_chunk_size = 0
                        budget = chunk_size

        with open(out_path, 'w', encoding='utf-8', newline='\n') as out_f:
            with multiprocessing.Pool(num_process) as pool:
                for i, (processed_lines, unfiltered_line_num) in \
                        enumerate(pool.imap(self.process_chunk, chunk_iterator())):
                    out_f.write('\n'.join(processed_lines) + '\n')
                    filtered_line_count += len(processed_lines)
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
        description='Clean mono corpus used in machine translation.')
    parser.add_argument('--corpus', type=str, nargs='+', required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the cleaned and tokenized corpus. If not set, '
                             'the default is "corpus.tok.{lang}"')
    parser.add_argument('--tokenizer', type=str, default='moses')
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
    corpus_processor = MonoCorpusProcessor(lang=args.lang,
                                           tokenizer=args.tokenizer,
                                           min_num_words=args.min_num_words,
                                           max_num_words=args.max_num_words,
                                           discard_non_latin1=args.discard_non_latin1)
    print('Clean the mono corpus:')
    print('   {}: {}'.format(args.lang, args.corpus))
    if args.save_path is None:
        save_path = 'corpus.tok.{}'.format(args.lang)
    else:
        save_path = args.save_path
    print('Save to {} -> {} \n'.format(args.lang, save_path))
    if os.path.exists(save_path) and not args.overwrite:
        warnings.warn('{} or {} exists, skip. If you need to overwrite this file, '
                      'rerun the script with --overwrite.'.format(save_path))
    else:
        corpus_processor.process_mono_corpus(
            corpus_paths=args.corpus,
            out_path=save_path,
            num_process=args.num_process)

def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
