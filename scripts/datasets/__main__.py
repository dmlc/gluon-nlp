import argparse
import importlib
import os

SUBCOMMAND_DICT = dict()

# Find all modules starting with `prepare_`
CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
for root, dirs, files in os.walk(CURR_DIR, topdown=False):
    for name in files:
        if name.startswith('prepare_') and name.endswith('.py'):
            command = name[:-3]
            path = os.path.join(root, name)
            relpath = os.path.relpath(path, CURR_DIR)[:-3]
            if relpath.startswith(os.sep):
                relpath = path[len(os.sep):]
            subpackage = relpath.replace(os.sep, '.')
            SUBCOMMAND_DICT[command] = 'gluonnlp.cli.data.' + subpackage


def cli_main():
    parser = argparse.ArgumentParser(
        description='Build-in scripts for downloading and preparing the data in GluonNLP.',
        prog='nlp_data', add_help=False)
    parser.add_argument('command', type=str,
                        choices=sorted(SUBCOMMAND_DICT.keys()) + ['help'],
                        metavar='[subcommand]',
                        help='The subcommand to use. '
                             'Choices are {}.'.format(sorted(SUBCOMMAND_DICT.keys()) + ['help']))
    args, other_args = parser.parse_known_args()
    if args.command == 'help':
        parser.print_help()
    else:
        mod = importlib.import_module(SUBCOMMAND_DICT[args.command])
        parser = mod.get_parser()
        sub_args = parser.parse_args(other_args)
        mod.main(sub_args)


if __name__ == '__main__':
    cli_main()
