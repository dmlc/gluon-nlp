import argparse

from .machine_translation import prepare_wmt
from .question_answering import prepare_squad
from .language_modeling import prepare_lm
from .music_generation import prepare_music_midi
from .pretrain_corpus import prepare_bookcorpus, prepare_wikipedia

SUBCOMMANDS = ['prepare_wmt', 'prepare_squad', 'prepare_lm',
               'prepare_music_midi', 'prepare_bookcorpus', 'prepare_wikipedia',
               'help']

def cli_main():
    parser = argparse.ArgumentParser(
        description='Build-in scripts for downloading and preparing the data in GluonNLP.',
        prog='nlp_data', add_help=False)
    parser.add_argument('command', type=str,
                        choices=SUBCOMMANDS,
                        metavar='[subcommand]',
                        help='The subcommand to use. '
                             'Choices are {}.'.format(SUBCOMMANDS))
    args, other_args = parser.parse_known_args()
    if args.command == 'prepare_wmt':
        parser = prepare_wmt.get_parser()
        sub_args = parser.parse_args(other_args)
        prepare_wmt.main(sub_args)
    elif args.command == 'prepare_squad':
        parser = prepare_squad.get_parser()
        sub_args = parser.parse_args(other_args)
        prepare_squad.main(sub_args)
    elif args.command == 'prepare_lm':
        parser = prepare_lm.get_parser()
        sub_args = parser.parse_args(other_args)
        prepare_lm.main(sub_args)
    elif args.command == 'prepare_music_midi':
        parser = prepare_music_midi.get_parser()
        sub_args = parser.parse_args(other_args)
        prepare_music_midi.main(sub_args)
    elif args.command == 'prepare_bookcorpus':
        parser = prepare_bookcorpus.get_parser()
        sub_args = parser.parse_args(other_args)
        prepare_bookcorpus.main(sub_args)
    elif args.command == 'prepare_wikipedia':
        parser = prepare_wikipedia.get_parser()
        sub_args = parser.parse_args(other_args)
        prepare_wikipedia.main(sub_args)
    elif args.command == 'help':
        parser.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    cli_main()
