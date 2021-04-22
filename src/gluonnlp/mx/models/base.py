__all__ = ['list_backbone_names', 'get_backbone', 'BACKBONE_REGISTRY']

from typing import Tuple, List
from ..base import get_model_zoo_home_dir
from ..data.tokenizers import BaseTokenizer
from ..utils.registry import Registry
from mxnet.gluon import Block

BACKBONE_REGISTRY = Registry('Backbone Models')


def list_backbone_names():
    all_keys = []
    for backbone_type in BACKBONE_REGISTRY.list_keys():
        all_keys.extend(BACKBONE_REGISTRY.get(backbone_type)[-1]())
    return all_keys


def get_backbone(model_name: str,
                 root: str = get_model_zoo_home_dir(),
                 **kwargs) -> Tuple['Block', str, BaseTokenizer, str, List]:
    """Get the backbone network

    Parameters
    ----------
    model_name
        The name of the pretrained model
    root
        Downloaded directory of the model zoo

    Returns
    -------
    model_cls
        The class to construct the backbone network
    cfg
        Path to the config file of the backbone
    tokenizer
        The tokenizer that is bound to the backbone model
    backbone_param_path
        The path to the pretrained backbone weights
    others
        The other items returned by the create function.
         Will be wrapped into a list

    Examples
    --------

    >>> from gluonnlp.models import get_backbone
    >>> model_cls, cfg, tokenizer, backbone_param_path, _ = get_backbone('google_en_cased_bert_base')
    >>> model = model_cls.from_cfg(cfg)
    >>> model.load_parameters(backbone_param_path)
    """
    model_cls, local_create_fn = None, None

    for backbone_type in BACKBONE_REGISTRY.list_keys():
        ele_model_cls, ele_local_create_fn, list_key_fn = BACKBONE_REGISTRY.get(backbone_type)
        if model_name in list_key_fn():
            model_cls = ele_model_cls
            local_create_fn = ele_local_create_fn
    if model_cls is None or local_create_fn is None:
        raise KeyError('The backbone model "{}" is not found! '
                       'Here are all available backbone models = {}'
                       .format(model_name,
                               list_backbone_names()))
    cfg, tokenizer, local_params_path, *others = local_create_fn(model_name=model_name, root=root,
                                                                 **kwargs)
    return model_cls, cfg, tokenizer, local_params_path, others
