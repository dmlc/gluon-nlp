import torch as th
import numpy as np

numpy_to_torch_dtype_dict = {
    np.dtype(np.bool): th.bool,
    np.dtype(np.uint8): th.uint8,
    np.dtype(np.int8): th.int8,
    np.dtype(np.int16): th.int16,
    np.dtype(np.int32): th.int32,
    np.dtype(np.int64): th.int64,
    np.dtype(np.float16): th.float16,
    np.dtype(np.float32): th.float32,
    np.dtype(np.float64): th.float64,
    np.dtype(np.complex64): th.complex64,
    np.dtype(np.complex128): th.complex128
}

torch_dtype_to_numpy_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}


def to_torch_dtype(dtype):
    """Convert the dtype to pytorch data type

    Parameters
    ----------
    dtype
        The input dtype

    Returns
    -------
    ret
        Converted dtype
    """
    if isinstance(dtype, th.dtype) or dtype is None:
        return dtype
    dtype = np.dtype(dtype)
    if dtype in numpy_to_torch_dtype_dict:
        return numpy_to_torch_dtype_dict[dtype]
    else:
        raise KeyError(f'dtype = {dtype} is not supported for conversion')


def to_numpy_dtype(dtype):
    """Convert the dtype to numpy dtype

    Parameters
    ----------
    dtype
        Input dtype

    Returns
    -------
    ret
        The converted dtype
    """
    if dtype is None:
        return None
    if dtype in torch_dtype_to_numpy_dict:
        return torch_dtype_to_numpy_dict[dtype]
    else:
        return np.dtype(dtype)


def share_parameters(source, target):
    """Share parameters recursively from source model to target model.

    For example, if you want ``dense1`` to share ``dense0``'s weights, you can do::

        dense0 = nn.Linear(20)
        dense1 = nn.Linear(20)
        share_parameters(dense0, dense)

    which equals to
        dense1.weight = dense0.weight
        dense1.bias = dense0.bias

    Parameters
    ----------
    source : nn.Module
    target : nn.Module
    """
    def _named_members(module, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules.

        Unlike upstream torch implementation, this implementation returns
        members that are known under multiple names, such as shared
        parameters.

        """
        modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None:
                    continue
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    source_names = set(n for n, p in _named_members(source, lambda m: m._parameters.items()))
    target_names = set(n for n, p in _named_members(target, lambda m: m._parameters.items()))
    if not source_names == target_names:
        raise ValueError(
            'Source and target modules do not have the same set of parameters. '
            f'The following parameters are missing from target: "{source_names - target_names}"'
            f'The following parameters are missing from source: "{target_names - source_names}"')

    for name in source_names:
        module_names = name.split('.')
        weight_name = module_names.pop()
        tmp_source, tmp_target = source, target
        for module_name in module_names:
            tmp_source = tmp_source._modules[module_name]
            tmp_target = tmp_target._modules[module_name]
        setattr(tmp_target, weight_name, getattr(tmp_source, weight_name))


def move_to(obj, device=None):
    """

    Parameters
    ----------
    obj
        Nested torch object
    device
        The target device

    Returns
    -------
    new_obj
        The objects that have been moved to device.
    """
    if th.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, (list, tuple)):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        if isinstance(obj, tuple):
            res = tuple(res)
        return res
    else:
        raise TypeError("Invalid type for move_to")
