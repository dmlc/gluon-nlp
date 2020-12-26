import argparse
import mxnet as mx
import os

mx.npx.set_np()


def get_parser():
    parser = argparse.ArgumentParser(description='Script to average the checkpoints')
    parser.add_argument('--checkpoints', type=str, required=True, nargs='+',
                        help='checkpoint file paths, supports two format, '
                        '--checkpoints folder/epoch*.params or --checkpoints folder/update*.param')
    parser.add_argument('--ids', type=int, required=False, nargs='+',
                        help='The IDs of the checkpoints.')
    parser.add_argument('--begin', type=int, required=False,
                        default=None,
                        help='begin number of checkpoints')
    parser.add_argument('--end', type=int, required=False,
                        default=None,
                        help='end number of checkpoints. '
                             'We select the checkpoints with ID >= begin and <= end.')
    parser.add_argument('--save-path', type=str, required=True, help='Path of the output file')
    return parser


def main(args):
    assert args.begin >= 0
    assert args.end >= args.begin
    args.range = list(range(args.begin, args.end + 1))
    if args.begin is not None or args.end is not None or args.ids is not None:
        print(f'Before filtering, the checkpoints are {args.checkpoints}')
        prefix = os.path.commonprefix(args.checkpoints)
        postfix = os.path.commonprefix([ele[::-1] for ele in args.checkpoints])[::-1]
        checkpoint_id_l = [int(ele[len(prefix):-len(postfix)]) for ele in args.checkpoints]
        ckpt_paths = []
        if args.ids is not None:
            for ele in args.ids:
                assert ele in checkpoint_id_l
                ckpt_paths.append(f'{prefix}{ele}{postfix}')
        else:
            assert args.begin is not None and args.end is not None, \
                'Must specify both begin and end if you want to select a range!'
            for ele in checkpoint_id_l:
                if ele >= args.begin and ele <= args.end:
                    ckpt_paths.append(f'{prefix}{ele}{postfix}')
    else:
        ckpt_paths = args.checkpoints
    print(f'Load models from {ckpt_paths}')
    print('Average the models and save it to {}'.format(args.save_path))
    assert len(ckpt_paths) > 0, 'Cannot found checkpoints. You may need to check the inputs again.'
    res = mx.npx.load(ckpt_paths[0])
    keys = res.keys()
    for ckpt_path in ckpt_paths[1:]:
        ckpt = mx.npx.load(ckpt_path)
        for key in keys:
            res[key] += ckpt[key]
    for key in keys:
        res[key] /= len(ckpt_paths)
    mx.npx.savez(args.save_path, **res)


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
