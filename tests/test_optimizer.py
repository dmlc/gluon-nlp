import itertools
import numpy as np
from gluonnlp.optimizer import AdamW
import mxnet as mx
from mxnet.test_utils import compare_optimizer
mx.npx.reset_np()


def test_adam():
    opt1 = AdamW
    opt2 = AdamW
    shapes = [(3, 4, 5), (10, 4), (7,)]
    beta1_options = [{}, {'beta1': 0.5}, {'beta1': 0.7}]
    beta2_options = [{}, {'beta2': 0.8}, {'beta2': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{'multi_precision': False}]  # TODO(sxjscience) Test for FP16
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    correct_bias_options = [{'correct_bias': True}, {'correct_bias': False}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(beta1_options, beta2_options, cg_options,
                                        rg_options, wd_options, mp_options,
                                        agg_options, correct_bias_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg), shapes, dtype,
                              rtol=1e-4, atol=2e-5)
