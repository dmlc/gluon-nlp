import argparse
import mxnet as mx
import re

mx.npx.set_np()

def get_parser():
    parser = argparse.ArgumentParser(description='Script to average the checkpoints')
    parser.add_argument('--checkpoints', type=str, required=True, nargs='+',
                        help='checkpoint file paths, supports two format, '
                        '--checkpoints folder/epoch*.params or --checkpoints folder/update*.param')
    parser.add_argument('--begin', type=int, required=True, help='begin number of checkpoints')
    parser.add_argument('--end', type=int, required=True, help='end number of checkpoints')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path of the output file')
    return parser

def main(args):
    assert args.begin >= 0
    assert args.end >= args.begin
    args.range = list(range(args.begin, args.end + 1))
    
    ckpt_epochs_regexp = re.compile(r'(.*\/)?epoch(\d+)\.params')
    ckpt_updates_regexp = re.compile(r'(.*\/)?update(\d+)\.params')
    ckpt_path = args.checkpoints[0]
    if ckpt_epochs_regexp.fullmatch(ckpt_path) is not None:
        ckpt_regexp = ckpt_epochs_regexp
    elif ckpt_updates_regexp.fullmatch(ckpt_path) is not None:
        ckpt_regexp = ckpt_updates_regexp
    else:
        raise Exception('Wrong checkpoints path format')
    
    ckpt_paths = []
    for path in args.checkpoints:
        m = ckpt_regexp.fullmatch(path)
        assert m is not None, 'Wrong checkpoints path format'
        num = int(m.group(2))
        if num >= args.begin and num <= args.end:
            ckpt_paths.append(path)
    
    assert len(ckpt_paths) > 0
    res = mx.npx.load(ckpt_paths[0])
    keys = res.keys()
    for ckpt_path in ckpt_paths[1:]:
        ckpt = mx.npx.load(ckpt_path)
        for key in keys:
            res[key] += ckpt[key]
    for key in keys:
        res[key] /= len(args.range)
    mx.npx.save(args.save_path, res)

def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
