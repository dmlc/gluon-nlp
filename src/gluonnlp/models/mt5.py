# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
MT5 Model

@misc{xue2020mt5,
    title = {{mT5}: A massively multilingual pre-trained text-to-text transformer},
    author = {Linting Xue and Noah Constant and Adam Roberts and Mihir Kale and Rami Al-Rfou and Aditya Siddhant and Aditya Barua and Colin Raffel},
    year = {2020},
    eprint = {2010.11934},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
"""


__all__ = ['MT5Model', 'MT5Inference']


import os
from typing import Tuple

from mxnet import use_np

from .base import BACKBONE_REGISTRY
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from .t5 import T5Encoder, T5Model, T5Inference, T5Tokenizer
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..utils.registry import Registry


FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'mt5.txt'))


mt5_cfg_reg = Registry('mt5_cfg')

@mt5_cfg_reg.register()
def google_mt5_base(): 
    """Configuration of mT5 Base"""
    cfg = CN()
    # model parameters
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 250112
    cfg.MODEL.d_model = 768
    cfg.MODEL.d_kv = 64
    cfg.MODEL.d_ff = 2048
    cfg.MODEL.num_layers = 12
    cfg.MODEL.num_heads = 12
    cfg.MODEL.dropout_prob = 0.1
    cfg.MODEL.layer_norm_eps = 1E-6
    cfg.MODEL.activation = 'gated-gelu'
    cfg.MODEL.dtype = 'float32'
    cfg.MODEL.layout = 'NT'
    # initializer parameters
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.init_factor = 1.0
    # other parameters
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


@mt5_cfg_reg.register()
def google_mt5_small(): 
    cfg = google_mt5_base()
    cfg.defrost()
    cfg.MODEL.d_model = 512
    cfg.MODEL.d_ff = 1024
    cfg.MODEL.num_layers = 8
    cfg.MODEL.num_heads = 6
    cfg.freeze()
    return cfg


@mt5_cfg_reg.register()
def google_mt5_large(): 
    cfg = google_mt5_base()
    cfg.defrost()
    cfg.MODEL.d_model = 1024
    cfg.MODEL.d_ff = 2816
    cfg.MODEL.num_layers = 24
    cfg.MODEL.num_heads = 16
    cfg.freeze()
    return cfg


@mt5_cfg_reg.register()
def google_mt5_xl(): 
    cfg = google_mt5_base()
    cfg.defrost()
    cfg.MODEL.d_model = 2048
    cfg.MODEL.d_ff = 5120
    cfg.MODEL.num_layers = 24
    cfg.MODEL.num_heads = 32
    cfg.freeze()
    return cfg


@mt5_cfg_reg.register()
def google_mt5_xxl(): 
    cfg = google_mt5_base()
    cfg.defrost()
    cfg.MODEL.d_model = 4096
    cfg.MODEL.d_ff = 10240
    cfg.MODEL.num_layers = 24
    cfg.MODEL.num_heads = 64
    cfg.freeze()
    return cfg


PRETRAINED_URL = {
    'google_mt5_small': {
        'cfg': google_mt5_small(), 
        'vocab': 'google_mt5_small/mt5-2730df74.vocab', 
        'params': 'google_mt5_small/model-b20e24d7.params'
    }, 
    'google_mt5_base': {
        'cfg': google_mt5_base(), 
        'vocab': 'google_mt5_base/mt5-2730df74.vocab', 
        'params': 'google_mt5_base/model-91eaa894.params'
    }, 
    'google_mt5_large': {
        'cfg': google_mt5_large(), 
        'vocab': 'google_mt5_large/mt5-2730df74.vocab', 
        'params': 'google_mt5_large/model-6b46e841.params'
    }, 
    'google_mt5_xl': {
        'cfg': google_mt5_xl(), 
        'vocab': 'google_mt5_xl/mt5-2730df74.vocab', 
        'params': 'google_mt5_xl/model-7655ea81.params'
    }, 
    'google_mt5_xxl': {
        'cfg': google_mt5_xxl(), 
        'vocab': 'google_mt5_xxl/mt5-2730df74.vocab', 
        'params': 'google_mt5_xxl/model-2e9e44b9.params'
    }
}


@use_np
class MT5Model(T5Model): 
    @classmethod
    def get_cfg(cls, key=None): 
        if key is None: 
            return google_mt5_base()
        else: 
            return mt5_cfg_reg.create(key)


@use_np
class MT5Inference(T5Inference): 
    pass


class MT5Tokenizer(T5Tokenizer): 
    pass


def list_pretrained_mt5(): 
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_mt5(
    model_name: str = 'google_mt5_base', 
    root: str = get_model_zoo_home_dir(), 
    load_backbone: bool = True, 
    load_lm: bool = False, 
    extra_ids: int = 100
) -> Tuple[CN, MT5Tokenizer, str, str]: 
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}.'.format(
        model_name, list_pretrained_mt5())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    if isinstance(cfg_path, CN):
        cfg = cfg_path
    else:
        cfg = None
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']

    local_paths = dict()
    download_jobs = [('vocab', vocab_path)]
    if cfg is None: 
        download_jobs.append(('cfg', cfg_path))
    for key, path in download_jobs: 
        local_paths[key] = download(
            url=get_repo_model_zoo_url() + path, 
            path=os.path.join(root, path), 
            sha1_hash=FILE_STATS[path]
        )
    if load_backbone: 
        local_params_path = download(
            url=get_repo_model_zoo_url() + params_path, 
            path=os.path.join(root, params_path), 
            sha1_hash=FILE_STATS[params_path]
        )
    else: 
        local_params_path = None
    # lm model has not been implemented
    local_lm_params_path = None
    tokenizer = MT5Tokenizer(local_paths['vocab'], extra_ids)
    if cfg is None: 
        cfg = MT5Model.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, local_lm_params_path


BACKBONE_REGISTRY.register(
    'mt5', 
    [MT5Model, get_pretrained_mt5, list_pretrained_mt5]
)
