import os
from gluonnlp.cli import average_checkpoint
from mxnet.gluon import nn
from numpy.testing import assert_allclose

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))

def test_avg_ckpt():
    try:
        average_checkpoint.cli_main()
    except:
        pass
    num_ckpts = 5
    model = nn.Dense(units=10, in_units=10)
    model.initialize()
    params = model.collect_params()
    gd_avg = {}
    for key in params.keys():
        gd_avg[key] = params[key].data().asnumpy()
    model.save_parameters(os.path.join(_CURR_DIR, 'update0.params'))
    
    for i in range(1, num_ckpts):
        model.initialize(force_reinit=True)
        params = model.collect_params()
        for key in gd_avg.keys():
            gd_avg[key] += params[key].data().asnumpy()
        model.save_parameters(os.path.join(_CURR_DIR, 'update{}.params'.format(i)))
    
    for key in gd_avg.keys():
        gd_avg[key] /= num_ckpts

    parser = average_checkpoint.get_parser()
    args = parser.parse_args(['--checkpoints', None,
                              '--begin', '0',
                              '--end', str(num_ckpts-1),
                              '--save-path', os.path.join(_CURR_DIR, 'avg.params')])
    args.checkpoints = ['fake', 'ckpt']
    try:
        average_checkpoint.main(args)
    except:
        pass
    args.checkpoints = [os.path.join(_CURR_DIR, 'update{}.params'.format(i)) \
                        for i in range(0, num_ckpts)]
    average_checkpoint.main(args)
    
    model.load_parameters(os.path.join(_CURR_DIR, 'avg.params'))
    params = model.collect_params()
    
    for key in gd_avg.keys():
        assert_allclose(gd_avg[key], params[key].data().asnumpy(), 1E-7, 1E-7)
