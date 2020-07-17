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
XLM-R Model

@article{conneau2019unsupervised,
  title={Unsupervised Cross-lingual Representation Learning at Scale},
  author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1911.02116},
  year={2019}
}
"""

__all__ = ['XLMRModel', 'XLMRForMLM', 'list_pretrained_xlmr', 'get_pretrained_xlmr']

from typing import Tuple
import os
from mxnet import use_np
from .roberta import RobertaModel, RobertaForMLM, roberta_base, roberta_large
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..utils.config import CfgNode as CN
from ..utils.registry import Registry
from ..utils.misc import load_checksum_stats, download
from ..registry import BACKBONE_REGISTRY
from ..data.tokenizers import SentencepieceTokenizer


PRETRAINED_URL = {
    'fairseq_xlmr_base': {
        'cfg': 'fairseq_xlmr_base/model-b893d178.yml',
        'sentencepiece.model': 'fairseq_xlmr_base/sentencepiece-18e17bae.model',
        'params': 'fairseq_xlmr_base/model-3fa134e9.params',
        'mlm_params': 'model_mlm-86e37954.params',
        'lowercase': False,
    },
    'fairseq_xlmr_large': {
        'cfg': 'fairseq_xlmr_large/model-01fc59fb.yml',
        'sentencepiece.model': 'fairseq_xlmr_large/sentencepiece-18e17bae.model',
        'params': 'fairseq_xlmr_large/model-b62b074c.params',
        'mlm_params': 'model_mlm-887506c2.params',
        'lowercase': False,
    }
}

FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'xlmr.txt'))
xlmr_cfg_reg = Registry('xlmr_cfg')


@xlmr_cfg_reg.register()
def xlmr_base():
    cfg = roberta_base()
    cfg.defrost()
    cfg.MODEL.vocab_size = 250002
    cfg.freeze()
    return cfg


@xlmr_cfg_reg.register()
def xlmr_large():
    cfg = roberta_large()
    cfg.defrost()
    cfg.MODEL.vocab_size = 250002
    cfg.freeze()
    return cfg


@use_np
class XLMRModel(RobertaModel):
    @staticmethod
    def get_cfg(key=None):
        if key:
            return xlmr_cfg_reg.create(key)
        else:
            return xlmr_base()
@use_np
class XLMRForMLM(RobertaForMLM):
    pass

def list_pretrained_xlmr():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_xlmr(model_name: str = 'fairseq_xlmr_base',
                        root: str = get_model_zoo_home_dir(),
                        load_backbone: bool = True,
                        load_mlm: bool = False) \
        -> Tuple[CN, SentencepieceTokenizer, str]:
    """Get the pretrained XLM-R weights

    Parameters
    ----------
    model_name
        The name of the xlmr model.
    root
        The downloading root
    load_backbone
        Whether to load the weights of the backbone network
    load_mlm
        Whether to load the weights of MLM

    Returns
    -------
    cfg
        Network configuration
    tokenizer
        The HuggingFaceByteBPETokenizer
    params_path
        Path to the parameters
    mlm_params_path
        Path to the parameter that includes both the backbone and the MLM
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_xlmr())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    sp_model_path = PRETRAINED_URL[model_name]['sentencepiece.model']
    params_path = PRETRAINED_URL[model_name]['params']
    mlm_params_path = PRETRAINED_URL[model_name]['mlm_params']
    local_paths = dict()
    for k, path in [('cfg', cfg_path), ('sentencepiece.model', sp_model_path)]:
        local_paths[k] = download(url=get_repo_model_zoo_url() + path,
                                  path=os.path.join(root, path),
                                  sha1_hash=FILE_STATS[path])
    if load_backbone:
        local_params_path = download(url=get_repo_model_zoo_url() + params_path,
                                     path=os.path.join(root, params_path),
                                     sha1_hash=FILE_STATS[params_path])
    else:
        local_params_path = None
    if load_mlm and mlm_params_path is not None:
        local_mlm_params_path = download(url=get_repo_model_zoo_url() + mlm_params_path,
                                         path=os.path.join(root, mlm_params_path),
                                         sha1_hash=FILE_STATS[mlm_params_path])
    else:
        local_mlm_params_path = None

    do_lower = True if 'lowercase' in PRETRAINED_URL[model_name]\
                       and PRETRAINED_URL[model_name]['lowercase'] else False
    tokenizer = SentencepieceTokenizer(
                    model_path=local_paths['sentencepiece.model'],
                    lowercase=do_lower)
    cfg = XLMRModel.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, local_mlm_params_path


BACKBONE_REGISTRY.register('xlmr', [XLMRModel,
                                    get_pretrained_xlmr,
                                    list_pretrained_xlmr])
