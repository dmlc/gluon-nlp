"""Convert pre-trained model parameters from ElectraForPretrain to ElectraModel"""

import os
import argparse
import mxnet as mx

from pretraining_utils import get_electra_pretraining_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model-name', type=str, default='google_electra_small',
                       help='Name of the pretrained model.')
    parser.add_argument('--params-file', type=str, required=True,
                        help='Path to the pretrained parameter file.')
    parser.add_argument('--out-file', type=str, default=None,
                        help='Output file path.')
    parser.add_argument('--generator_units_scale', type=float, default=None,
                        help='The scale size of the generator units, same as used in pretraining.')
    parser.add_argument('--generator_layers_scale', type=float, default=None,
                        help='The scale size of the generator layer, same as used in pretraining.')

    args = parser.parse_args()
    return args


def convert_params(model_name, generator_units_scale, generator_layers_scale,
                   params_path, out_path):
    _, _, pretrain_model = get_electra_pretraining_model(model_name, [mx.cpu()],
                                                         generator_units_scale=generator_units_scale,
                                                         generator_layers_scale=generator_layers_scale,
                                                         params_path=params_path)
    backbone_model = pretrain_model.disc_backbone
    backbone_model.save_parameters(out_path)


if __name__ == '__main__':
    args = parse_args()
    out_path = args.out_file
    if not out_path:
        params_file = args.params_file
        file_name_sep = os.path.basename(params_file).split(os.path.extsep)
        file_name_sep.insert(-1, 'backbone')
        out_path = os.path.join(
            os.path.dirname(params_file),
            os.path.extsep.join(file_name_sep))
    convert_params(args.model_name, args.generator_units_scale, args.generator_layers_scale,
                   args.params_file, out_path)
