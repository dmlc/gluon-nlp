__all__ = ['get_ec2_tvm_flags', 'update_tvm_convert_map']

import tvm.relay.op as _op
import tvm.relay.expr as _expr
from typing import Dict
from tvm.relay.frontend.mxnet import _convert_map
from tvm.relay.frontend.common import infer_type as _infer_type

def get_ec2_tvm_flags() -> Dict[str, Dict]:
    r"""Return the recommended flags for TVM compilation in AWS EC2 instances.

    Including C4, C5, G4, P3.

    For more details about AWS EC2 instances, refer to https://aws.amazon.com/ec2/instance-types/.

    Returns
    -------
    info_dict
        A dictionary that contains the mapping between instance type and the
        corresponding compilation flags.
        Each element includes:

        - target
            The compilation target
        - use_gpu
            Whether it's a GPU instance
        - opt_level
            The optimization level in compilation
        - pass
            Additional graph passes for further improvement.
    """
    instance_info = {
        'g4': {'target': "cuda -model=t4 -libs=cublas,cudnn",
               'use_gpu': True,
               'opt_level': 3,
               'required_pass': ["FastMath"]},
        'c4': {'target': 'llvm -mcpu=core-avx2 -libs=cblas',
               'use_gpu': False,
               'opt_level': 3,
               'required_pass': ["FastMath"]},
        'c5': {'target': 'llvm -mcpu=skylake-avx512 -libs=cblas',
               'use_gpu': False,
               'opt_level': 3,
               'required_pass': ["FastMath"]},
        'p3': {'target': 'cuda -model=v100 -libs=cublas,cudnn',
               'use_gpu': True,
               'opt_level': 3,
               'required_pass': ["FastMath"]}
    }
    return instance_info


def update_tvm_convert_map() -> None:
    op = (('masked_softmax', _mx_masked_softmax))
    _convert_map.update({key: value for key, value in op})


def _mx_masked_softmax(inputs, attrs):
    assert len(inputs) == 1 or len(inputs) == 2
    axis = attrs.get_int("axis")
    temperature = attrs.get_float("temperature")
    if len(inputs) == 1:
        result = _op.nn.softmax(inputs[0] / _expr.const(temperature), axis=axis)
    else:
        neg = -1e18
        att_score, mask = inputs
        att_score_dtype = _infer_type(att_score).checked_type.dtype
        if att_score_dtype == "float16":
            neg = -1e4
        temp = _op.where(mask, 
                         att_score,
                         _expr.const(neg))
        result = _op.multiply(_op.nn.softmax(temp / _expr.const(temperature), axis=axis), mask.astype("float32"))
    return result
