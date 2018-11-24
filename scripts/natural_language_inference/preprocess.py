"""
Tokenize SNLI data.
"""

import argparse
import csv
import nltk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
            help='.txt file for the SNLI dataset')
    parser.add_argument('--output',
            help='path for tokenized output file')
    args = parser.parse_args()
    return args

def read_tokens(tree_str):
    t = nltk.Tree.fromstring(tree_str)
    return t.leaves()

def main(args):
    examples = []
    with open(args.input, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t')
        for cols in reader:
            s1 = read_tokens(cols['sentence1_parse'])
            s2 = read_tokens(cols['sentence2_parse'])
            label = cols['gold_label']
            if label in ('neutral', 'contradiction', 'entailment'):
                examples.append((s1, s2, label))
    with open(args.output, 'w') as fout:
        for s1, s2, l in examples:
            fout.write('{}\t{}\t{}\n'.format(' '.join(s1), ' '.join(s2), l))


if __name__ == '__main__':
    args = parse_args()
    main(args)
