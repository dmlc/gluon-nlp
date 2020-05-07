import argparse
import textwrap
from multiprocessing import Pool
import numpy as np
import time
from gluonnlp.data import tokenizers

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
    Encode based on different implementations.

    We support the following models:

        "python apply_subword.py --model spm" : Encode with Sentencepiece Model;
        "python apply_subword.py --model subword_nmt" : Encode with the subword-nmt package;
        "python apply_subword.py --model yttm" : Encode with YouTokenToMe; 
        "python apply_subword.py --model hf_bytebpe" : Encode with the Byte-level BPE Tokenizer Implemented by Huggingface.
        "python apply_subword.py --model hf_wordpiece" : Encode with the Wordpiece Tokenizer Implementated by Huggingface.
        "python apply_subword.py --model hf_bpe" : Encode with the BPE Tokenizer Implemented by Huggingface.
    ''')
    )
    parser.add_argument('--corpus', type=str, nargs='+', required=True,
                        help='Path of the corpus. '
                             'You may input multiple corpus files separated by space.')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path of the output file')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path of the model file')
    parser.add_argument('--vocab-path', type=str, default=None,
                        help='Path of the vocabulary file')
    parser.add_argument('--model', type=str, choices=['spm',
                                                      'subword_nmt',
                                                      'yttm',
                                                      'hf_bytebpe',
                                                      'hf_wordpiece',
                                                      'hf_bpe'],
                        required=True, help='Subword model type')
    parser.add_argument('--num-process', type=int, default=16,
                        help='Number of process')
    parser.add_argument('--lowercase', action='store_true', default=False,
                        help='Use lowercase, '
                        'only applicable to hf_bpe, hf_bytebpe and hf_wordpiece')
    parser.add_argument('--strip-accents', action='store_true', default=False,
                        help='Disable BERT characters normalization, '
                        'only applicable to hf_wordpiece')
    parser.add_argument('--output-type', type=str, choices=['subword', 'id'], default='subword',
                        help='Whether output subwords or ids')
    parser.add_argument('--bpe-dropout', type=float, default=None,
                        help='BPE dropout, applicable to subword_nmt, yttm, hf_bpe and hf_bytebpe')
    
    return parser

class ParallelCorpusApplyer:
    def __init__(self, corpus, tokenizer_model, output_type):
        self.chunk_size = 1024 * 1024
        self.corpus = corpus
        self.tokenizer_model = tokenizer_model
        self.output_type = output_type
        
    def chunk_iter(self, step=10):
        for corpus_path in self.corpus:
            line_pos = [0]
            with open(corpus_path, 'rb') as fcb:
                pos = 0
                for line in fcb:
                    pos += len(line)
                    line_pos.append(pos)
            line_pos = np.array(line_pos, dtype=np.int64)
            line_size = line_pos[1:] - line_pos[:-1]
            num_lines = line_pos.shape[0] - 1
            budget = self.chunk_size
            chunk_start = 0
            cur_chunk_size = 0
            for i in range(0, num_lines, step):
                line_batch_num = min(num_lines - i, step)
                batch_line_size = line_size[i:(i + line_batch_num)].sum()
                budget -= batch_line_size
                cur_chunk_size += batch_line_size
                if budget <= 0 or i + step >= num_lines:
                    yield corpus_path, chunk_start, cur_chunk_size
                    chunk_start += cur_chunk_size
                    budget = self.chunk_size
                    cur_chunk_size = 0
        
    def process_chunk(self, args):
        corpus_path, chunk_start, cur_chunk_size = args
        with open(corpus_path, 'rb') as fcb:
            fcb.seek(chunk_start)
            lines_byte = fcb.read(cur_chunk_size)
            lines_byte = lines_byte.splitlines()
            sentences = [line_byte.decode('utf-8').strip() for line_byte in lines_byte]
            all_tokens = self.tokenizer_model.encode(sentences, self.output_type)
            tokenized_sentences = []
            for ele_tokens in all_tokens:
                if self.output_type == int:
                    ele_tokens = [str(token) for token in ele_tokens]
                tokenized_sentences.append(' '.join(ele_tokens))
            return tokenized_sentences

def main(args):
    start = time.time()
    if args.model == 'spm':
        tokenizer_model = tokenizers.create('spm',
                                            model_path=args.model_path,
                                            vocab=args.vocab_path)
    elif args.model == 'subword_nmt':
        tokenizer_model = tokenizers.create('subword_nmt',
                                            codec_path=args.model_path,
                                            vocab_path=args.vocab_path,
                                            bpe_dropout=args.bpe_dropout)
    elif args.model == 'yttm':
        args.bpe_dropout = 0.0 if not args.bpe_dropout else args.bpe_dropout
        tokenizer_model = tokenizers.create('yttm',
                                            model_path=args.model_path,
                                            bpe_dropout=args.bpe_dropout,
                                            n_threads=1)
    elif args.model == 'hf_bytebpe':
        tokenizer_model = tokenizers.create('hf_bytebpe',
                                            merges_file=args.model_path,
                                            vocab_file=args.vocab_path,
                                            dropout=args.bpe_dropout,
                                            lowercase=args.lowercase)
    elif args.model == 'hf_wordpiece':
        tokenizer_model = tokenizers.create('hf_wordpiece',
                                            vocab_file=args.vocab_path,
                                            lowercase=args.lowercase,
                                            strip_accents=args.strip_accents)
    elif args.model == 'hf_bpe':
        tokenizer_model = tokenizers.create('hf_bpe',
                                            merges_file=args.model_path,
                                            vocab_file=args.vocab_path,
                                            dropout=args.bpe_dropout,
                                            lowercase=args.lowercase)
    else:
        raise NotImplementedError
    print('Applying {} to {}'. format(tokenizer_model.__class__.__name__,
                                      ', '.join(args.corpus)))
    output_type = {'subword': str, 'id': int}[args.output_type]
    applyer = ParallelCorpusApplyer(args.corpus, tokenizer_model, output_type)
    with open(args.save_path, 'w', encoding='utf-8', newline='\n') as fo:
        with Pool(args.num_process) as pool:
            for i, tokenized_sentences in \
                enumerate(pool.imap(applyer.process_chunk, applyer.chunk_iter())):
                fo.write('\n'.join(tokenized_sentences))
                fo.write('\n')
    end = time.time()
    print('Done, Time spent {}\n'.format(end - start))

def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    cli_main()
