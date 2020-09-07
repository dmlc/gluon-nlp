import re

import argparse
import sacrebleu


def parse_args():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--generated_file', type=str, required=True, help='Model name')
    return parser.parse_args()


def calculate_self_bleu4(samples):
    pass

def calculate_zipf_coefficient():
    pass

def calculate_repetition():
    pass

def calculate_metrics(args):
    with open(args.generated_file, encoding='utf-8') as of:
        samples = of.read()
    pattern = '='*40 + ' SAMPLE \d+ ' + '='*40 + '\n'
    samples = re.split(pattern, samples)[1:]
    
    # self bleu4
    self_bleu4 = calculate_self_bleu4(samples)
    
    # zipf coefficient
    zipf_coefficient = calculate_zipf_coefficient()
    
    # repetition
    repetition = calculate_repetition()
    
    print()

if __name__ == '__main__':
    args = parse_args()
    calculate_metrics(args)
    