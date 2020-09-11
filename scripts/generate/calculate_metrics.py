import re
import argparse
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
import operator
import numpy as np
from scipy import stats
import random
from gluonnlp.models.gpt2 import get_pretrained_gpt2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate metrics for the generated sentences')
    parser.add_argument('--file', type=str, required=True, help='Model name')
    parser.add_argument('--num_samples', type=int, default=1000, help='')
    parser.add_argument('--num_bleu_samples', type=int, default=1000, help='')
    return parser.parse_args()


def calculate_self_bleu4(samples, num_bleu_samples):
    """Self- BLEU is calculated by computing the BLEU score of each generated document
    using all other generations in the evaluation set as references.
    """
    sys_indices = random.sample(range(len(samples)), num_bleu_samples)
    res = 0
    for sys_indice in sys_indices:
        res += sentence_bleu(
            hypothesis=samples[sys_indice],
            references=samples[:sys_indice] + samples[sys_indice+1:],
            weights=(0.25, 0.25, 0.25, 0.25))
    res /= len(samples)
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
    """The repetition rate in generated samples.
    """
    max_n = 90
    n_repeated_examples = 0
    for sample_id in sample_ids:
        rev = list(reversed(sample_id))
        last_n_repeats = [0 for _ in range(max_n)]
        for n in range(1, max_n + 1):
            n_repeat = 1
            while len(rev[n*n_repeat:n*(n_repeat+1)]) == n and \
                  rev[n*n_repeat:n*(n_repeat+1)] == rev[:n]:
                n_repeat += 1
            last_n_repeats[n-1] = n_repeat
        max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])
        if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n+1 >= 3 or last_n_repeats[max_repeated_n] > 50):
            n_repeated_examples += 1
    return n_repeated_examples / len(sample_ids)


def calculate_metrics(args):
    with open(args.generated_file, encoding='utf-8') as of:
        samples = of.read()
    pattern = '='*40 + ' SAMPLE \d+ ' + '='*40 + '\n'
    samples = re.split(pattern, samples)[1:]
    samples = samples[:args.num_samples]
    assert len(samples) == args.num_samples
    
    _, _, tokenizer, _, _ = get_pretrained_gpt2(
        load_backbone=False,
        load_lm=False)
    sample_ids = tokenizer.encode(samples, output_type=int)

    self_bleu4 = calculate_self_bleu4(samples, args.num_bleu_samples)
    zipf_coefficient = calculate_zipf_coefficient(sample_ids, tokenizer)
    repetition = calculate_repetition(sample_ids)
    print('Self BLEU 4: {}\n'
          'Zipf coefficient: {}\n'
          'Repectition: {}\n'
          .format(self_bleu4, zipf_coefficient, repetition))


if __name__ == '__main__':
    args = parse_args()
    calculate_metrics(args)
