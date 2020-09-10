import re
import argparse
import sacrebleu
from collections import Counter
import operator
import numpy as np
from scipy import stats
import os
import random
from gluonnlp.base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from gluonnlp.utils.misc import load_checksum_stats, download
from gluonnlp.data.tokenizers import HuggingFaceByteBPETokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate metrics for the generated sentences')
    parser.add_argument('--file', type=str, required=True, help='Model name')
    parser.add_argument('--num_samples', type=int, default=1000, help='')
    parser.add_argument('--num_bleu_samples', type=int, default=1000, help='')
    return parser.parse_args()


def calculate_self_bleu4(sample_ids, num_bleu_samples):
    """Self- BLEU is calculated by computing the BLEU score of each generated document
    using all other generations in the evaluation set as references.
    """
    sys_indices = random.sample(range(len(sample_ids)), num_bleu_samples)
    res = 0
    for sys_indice in sys_indices:
        # remove it self
        ref = sample_ids[:sys_indice] + sample_ids[sys_indice+1:]
        sacrebleu_out = sacrebleu.corpus_bleu(
            sys_stream=sample_ids[sys_indice],
            ref_streams=ref)
        res += sacrebleu_out.score
    res /= len(sample_ids)
    return res


def calculate_zipf_coefficient(sample_ids, tokenizer):
    """The Zipfian coefficient s can be used to compare the distribution in a given
    text to a theoretically perfect exponential curve.
    """
    cnt = Counter()
    for sample_id in sample_ids:
        cnt.update(sample_id)
    
    xs = np.arange(1, min(len(cnt), len(tokenizer.vocab)))
    ys = np.array(sorted(cnt.values(), key=operator.neg)[:len(tokenizer.vocab)])
    _, _, r, _, _ = stats.linregress(np.log(xs), np.log(ys))
    return r


def calculate_repetition(sample_ids):
    """
    """
    max_n = 90
    res = 0
    for sample_id in sample_ids:
        rev = list(reversed(sample_id))
        last_n_repeats = [0 for _ in range(max_n)]
        for n in range(1, max_n + 1):
            n_repeat = 1
            while len(rev[n*n_repeat:n*(n_repeat+1)]) == n and \
                  rev[n*n_repeat:n*(n_repeat+1)] == rev[:n]:
                n_repeat += 1
            last_n_repeat[n-1] = n_repeat
#        res += (sum(last_n_repeat) / ) TODO
        

def calculate_metrics(args):
    with open(args.generated_file, encoding='utf-8') as of:
        samples = of.read()
    pattern = '='*40 + ' SAMPLE \d+ ' + '='*40 + '\n'
    samples = re.split(pattern, samples)[1:]
    samples = samples[:args.num_samples]
    assert len(samples) == args.num_samples
    
    local_paths = {}
    download_jobs = [('vocab', 'gpt2_124M/gpt2-9dc62091.vocab'),
                     ('merges', 'gpt2_124M/gpt2-396d4d8e.merges')]
    FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'gpt2.txt'))
    for k, path in download_jobs:
        local_paths[k] = download(url=get_repo_model_zoo_url() + path,
                                  path=os.path.join(get_model_zoo_home_dir(), path),
                                  sha1_hash=FILE_STATS[path])
    tokenizer = HuggingFaceByteBPETokenizer(
        merges_file=local_paths['merges'],
        vocab_file=local_paths['vocab'])
    sample_ids = tokenizer.encode(samples, output_type=int)

    self_bleu4 = calculate_self_bleu4(sample_ids, args.num_bleu_samples)
    zipf_coefficient = calculate_zipf_coefficient(sample_ids, tokenizer)
    repetition = calculate_repetition(sample_ids)
    print('Self BLEU 4: {}\n'
          'Zipf coefficient: {}\n'
          'Repectition: {}\n'
          .format(self_bleu4, zipf_coefficient, repetition))


if __name__ == '__main__':
    args = parse_args()
    calculate_metrics(args)
