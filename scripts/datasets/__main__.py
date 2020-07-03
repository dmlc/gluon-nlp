import argparse
from .machine_translation import prepare_wmt
from .question_answering import prepare_squad, prepare_hotpotqa, prepare_searchqa, prepare_triviaqa
from .language_modeling import prepare_lm
from .music_generation import prepare_music_midi
from .pretrain_corpus import prepare_bookcorpus, prepare_wikipedia, prepare_openwebtext
from .general_nlp_benchmark import prepare_glue
from gluonnlp.registry import DATA_PARSER_REGISTRY, DATA_MAIN_REGISTRY

# TODO(zheyuye), lazy_import theses data parser functions and data main function
# and their dependencies by a dictionary mapping the datasets names to the functions.
def list_all_subcommands():
    out = []
    for key in DATA_PARSER_REGISTRY.list_keys():
        if key not in DATA_MAIN_REGISTRY._obj_map:
            raise KeyError('The data cli "{}" is registered in parser but is missing'
                           ' in main'.format(key))
        out.append(key)
    return out


def cli_main():
    parser = argparse.ArgumentParser(
        description='Build-in scripts for downloading and preparing the data in GluonNLP.',
        prog='nlp_data', add_help=False)
    parser.add_argument('command', type=str,
                        choices=list_all_subcommands() + ['help'],
                        metavar='[subcommand]',
                        help='The subcommand to use. '
                             'Choices are {}.'.format(list_all_subcommands() + ['help']))
    args, other_args = parser.parse_known_args()
    if args.command == 'help':
        parser.print_help()
    else:
        parser = DATA_PARSER_REGISTRY.create(args.command)
        sub_args = parser.parse_args(other_args)
        DATA_MAIN_REGISTRY.create(args.command, sub_args)


if __name__ == '__main__':
    cli_main()
