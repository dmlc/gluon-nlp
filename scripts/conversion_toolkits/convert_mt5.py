import argparse
import logging
import os

from gluonnlp.utils.misc import logging_config, sha1sum
from gluonnlp.models.mt5 import MT5Model as Gluon_mT5
from transformers import MT5Model as HF_mT5

from convert_t5 import (
    parse_args, convert_config, convert_vocab, convert_params, rename, test_conversion
)


# these mappings come from https://huggingface.co/google
MT5_PRETRAINED_MODEL_MAP = {
    'google/mt5-small': 'google_mt5_small', 
    'google/mt5-base': 'google_mt5_base', 
    'google/mt5-large': 'google_mt5_large', 
    'google/mt5-xl': 'google_mt5_xl', 
    'google/mt5-xxl': 'google_mt5_xxl'
}

MT5_PRETRAINED_CONFIG_MAP = {
    'google/mt5-small': 'https://huggingface.co/google/mt5-small/raw/main/config.json', 
    'google/mt5-base': 'https://huggingface.co/google/mt5-base/raw/main/config.json', 
    'google/mt5-large': 'https://huggingface.co/google/mt5-large/raw/main/config.json', 
    'google/mt5-xl': 'https://huggingface.co/google/mt5-xl/raw/main/config.json', 
    'google/mt5-xxl': 'https://huggingface.co/google/mt5-xxl/raw/main/config.json'
}

MT5_PRETRAINED_VOCAB_MAP = {
    'google/mt5-small': 'https://huggingface.co/google/mt5-small/raw/main/spiece.model', 
    'google/mt5-base': 'https://huggingface.co/google/mt5-base/raw/main/spiece.model', 
    'google/mt5-large': 'https://huggingface.co/google/mt5-large/raw/main/spiece.model', 
    'google/mt5-xl': 'https://huggingface.co/google/mt5-xl/raw/main/spiece.model', 
    'google/mt5-xxl': 'https://huggingface.co/google/mt5-xxl/raw/main/spiece.model'
}


def parse_args(): 
    parser = argparse.ArgumentParser('Convert Huggingface mT5 Model to GluonNLP')
    parser.add_argument(
        'model_name', choices=list(MT5_PRETRAINED_MODEL_MAP.keys()), help='Name of pretrained T5 model in Huggingface.'
    )
    parser.add_argument(
        'dest_dir', help='Directory to save converted config, vocab and weights.'
    )
    parser.add_argument(
        '--test', action='store_true', required=False, default=False, help='Whether to test conversion correctness.'
    )
    args = parser.parse_args()
    # further process mappings
    args.tgt_model_name = MT5_PRETRAINED_MODEL_MAP[args.model_name]
    args.config_url = MT5_PRETRAINED_CONFIG_MAP[args.model_name]
    args.vocab_url = MT5_PRETRAINED_VOCAB_MAP[args.model_name]
    return args


def convert_mt5(args): 
    logging.info('converting mT5 model from Huggingface...')
    if not os.path.exists(args.dest_dir): 
        os.mkdir(args.dest_dir)
    converted = {}
    # convert and save vocab
    convert_vocab(args, converted)
    # convert and save config
    gluon_cfg = Gluon_mT5.get_cfg(args.tgt_model_name)
    gluon_cfg = convert_config(args, gluon_cfg, converted)
    # convert, (test), and save model
    hf_mt5 = HF_mT5.from_pretrained(args.model_name)
    gluon_mt5 = Gluon_mT5.from_cfg(gluon_cfg)
    gluon_mt5 = convert_params(args, converted, hf_mt5, gluon_mt5)
    gluon_mt5.hybridize()
    # test model if needed
    if args.test: 
        test_conversion(args, hf_mt5, gluon_mt5)
    # rename with sha1sum
    rename(args, converted)
    logging.info('conversion completed.')
    logging.info('file statistics:')
    for item, new_path in converted.items(): 
        logging.info('filename: {}\tsize: {}\tsha1sum: {}'.format(
            os.path.basename(new_path), os.path.getsize(new_path), sha1sum(new_path)
        ))
    return converted


if __name__ == "__main__": 
    args = parse_args()
    logging_config()
    convert_mt5(args)
