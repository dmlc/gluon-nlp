import argparse
import mxnet as mx

mx.npx.set_np()

def get_parser():
    parser = argparse.ArgumentParser(description='Script to average the checkpoints')
    parser.add_argument('--prefix', type=str, required=True,
                        help='Prefix of the checkpoint files, e.g. --prefix ckpt_folder/checkpoint')
    parser.add_argument('--suffix', type=str, default='',
                        help='Prefix of the checkpoint files, e.g. --suffix .params')
    parser.add_argument('--range', type=str, required=True,
                        help='a piece of python code to represent the checkpoint range, '
                             'e.g. [1,3,7] range(10) range(4000,5000,2) [5,6]+[7,8]')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path of the output file')
    return parser

def main(args):
    args.range = eval(args.range)
    assert isinstance(args.range, list)
    
    ckpt_paths = [args.prefix + str(i) + args.suffix for i in args.range]
    nums = len(ckpt_paths)
    
    res = mx.npx.load(ckpt_paths[0])
    keys = res.keys()
    for ckpt_path in ckpt_paths[1:]:
        ckpt = mx.npx.load(ckpt_path)
        for key in keys:
            res[key] += ckpt[key]
    for key in keys:
        res[key] /= nums
    mx.npx.save(args.save_path, res)

def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
