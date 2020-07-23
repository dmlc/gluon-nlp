import argparse
import mxnet as mx

mx.npx.set_np()

def get_parser():
    parser = argparse.ArgumentParser(description='Script to average the checkpoints')
    parser.add_argument('--checkpoints', type=str, required=True,
                        help='path of checkpoints, use * to represent the numbers, '
                        'e.g. --checkpoints folder/epoch*.prams')
    parser.add_argument('--range', type=str, nargs='+', required=True,
                        help='number of checkpoints, supports range and list format at present, '
                        'e.g. --range range(3) [4,7, 5] range(8,100,2)')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path of the output file')
    return parser

def main(args):
    temp_range = []
    try:
        for r in args.range:
            if len(r) > 5 and r[:5] == 'range':
                r = r[5:].strip()[1:-1].split(',')
                r = tuple([int(n.strip()) for n in r])
                assert len(r) >= 1 and len(r) <= 3
                temp_range.extend(range(*r))
            elif r[0] == '[' and r[-1] == ']':
                r = r[1:-1].split(',')
                r = [int(n.strip()) for n in r]
                temp_range.extend(r)
            else:
                raise NotImplementedError
    except:
        raise Exception('wrong range format')
    args.range = temp_range
    ckpt_paths = [args.checkpoints.replace('*', str(i)) for i in args.range]
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
