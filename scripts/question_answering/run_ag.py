import os
import mxnet as mx
import logging
from gluonnlp.utils.misc import logging_config, set_seed, parse_ctx
import autogluon as ag
from run_squad import parse_args, train, evaluate

default_args = parse_args()


@ag.args(
    lr=ag.space.Real(2e-5, 5e-5, log=True),
    n_epoch=3.0,
    max_grad_norm=ag.space.Real(0, 0.5),
)
def ag_train(args, reporter):
    mx.npx.set_np()
    logging.info('lr: {}, max_grad_norm: {}'.format(args.lr, args.max_grad_norm))
    # update the hyperparameters
    default_args.lr = args.lr
    default_args.max_grad_norm = args.max_grad_norm

    params_saved = train(default_args)
    default_args.params_saved = params_saved
    out_eval = evaluate(default_args, is_save=False)
    results = (out_eval['best_exact'] + out_eval['best_f1']) * 0.5
    reporter(epoch=args.n_epoch, accuracy=results, lr=args.lr, max_grad_norm=args.max_grad_norm)


def main():
    gpu_num = len(parse_ctx(default_args.gpus))
    myscheduler = ag.scheduler.HyperbandScheduler(ag_train,
                                                  resource={'num_cpus': 8, 'num_gpus': gpu_num},
                                                  num_trials=8,
                                                  time_attr='epoch',
                                                  reward_attr="accuracy")
    logging.info(myscheduler)
    myscheduler.run()
    myscheduler.join_tasks()
    logging.info('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                          myscheduler.get_best_reward()))


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    os.environ['MXNET_USE_FUSION'] = '0'  # Manually disable pointwise fusion
    logging_config(default_args.output_dir, name='autogluon')
    set_seed(default_args.seed)
    main()
