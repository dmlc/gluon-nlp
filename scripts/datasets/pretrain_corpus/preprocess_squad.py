import os
import json
from tqdm import tqdm
import argparse

from gluonnlp.data.tokenizers import SentencepieceTokenizer


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True,
                        help="path to extraed squad dataset")
    parser.add_argument("-o", "--output", default="preprocessed_squad",
                        help="directory for preprocessed features")
    parser.add_argument("--vocab_file", default="vocab-c3b41053.json",
                        help="vocabulary file of SentencepieceTokenizer")
    parser.add_argument("--spm_model", default="spm-65999e5d.model",
                        help="vocabulary file of SentencepieceTokenizer")
    return parser

def main(args):
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    fnames = os.listdir(args.input)
    fnames = [os.path.join(args.input, name) for name in fnames if name.endswith(".json")]
    with open(os.path.join(args.output, "squad.txt"), "w") as fp:
        for json_file in fnames:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for entry in tqdm(data['data']):
                title = entry['title']
                for paragraph in entry['paragraphs']:
                    context_text = paragraph['context'].strip()
                    fp.write(context_text+'\n')
                    for qa in paragraph['qas']:
                        query_text = qa['question'].strip()
                        if query_text:
                            fp.write(query_text+'\n')
                    # Two extra line break to mark the document separation
                    fp.write('\n\n')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
