import argparse
import textwrap

from . import (
    clean_tok_corpus,
    learn_subword,
    apply_subword
)


SUBCOMMANDS = ['clean_tok_para_corpus', 'clean_tok_mono_corpus',
               'learn_subword', 'apply_subword', 'help']


def cli_main():
    parser = argparse.ArgumentParser(
        description='Sharable data preprocessing utilities in GluonNLP.',
        prog='nlp_process', add_help=False)
    parser.add_argument('command', type=str,
                        choices=SUBCOMMANDS,
                        metavar='[subcommand]',
                        help='The subcommand to use. '
                             'Choices are {}.'.format(SUBCOMMANDS))
    args, other_args = parser.parse_known_args()
    if args.command == 'clean_tok_para_corpus':
        parser = clean_tok_corpus.get_parser.para()
        sub_args = parser.parse_args(other_args)
        clean_tok_corpus.main_para(sub_args)
    elif args.command == 'clean_tok_mono_corpus':
        parser = clean_tok_corpus.get_parser.mono()
        sub_args = parser.parse_args(other_args)
        clean_tok_corpus.main_mono(sub_args)
    elif args.command == 'learn_subword':
        parser = learn_subword.get_parser()
        sub_args = parser.parse_args(other_args)
        learn_subword.main(sub_args)
    elif args.command == 'apply_subword':
        parser = apply_subword.get_parser()
        sub_args = parser.parse_args(other_args)
        apply_subword.main(sub_args)
    elif args.command == 'help':
        parser.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    cli_main()
