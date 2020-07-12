import argparse

from . import (
    average_checkpoint
)

SUBCOMMANDS = ['average_checkpoint', 'help']

def cli_main():
    parser = argparse.ArgumentParser(
        description='Machine translation utilities in GluonNLP.',
        prog='nlp_nmt', add_help=False)
    parser.add_argument('command', type=str,
                        choices=SUBCOMMANDS,
                        metavar='[subcommand]',
                        help='The subcommand to use. '
                             'Choices are {}.'.format(SUBCOMMANDS))
    args, other_args = parser.parse_known_args()
    if args.command == 'average_checkpoint':
        parser = average_checkpoint.get_parser()
        sub_args = parser.parse_args(other_args)
        average_checkpoint.main(sub_args)
    elif args.command == 'help':
        parser.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    cli_main()
