import gluonnlp
import numpy as np
import mxnet as mx
import pandas as pd
import matplotlib.pyplot as plt
from gluonnlp.data.sampler import SplitSampler
from tqdm import tqdm
from mxnet.gluon import nn
from gluonnlp.models import get_backbone
from gluonnlp.utils.parameter import clip_grad_global_norm
from gluonnlp.utils.preprocessing import get_trimmed_lengths
from gluonnlp.utils.misc import get_mxnet_visible_ctx, grouper, repeat
from mxnet.gluon.data import batchify as bf
from mxnet.gluon.data import DataLoader
from mxnet.lr_scheduler import PolyScheduler
from gluonnlp.utils import set_seed

class TextPredictionNet(nn.HybridBlock):
    def __init__(self, backbone, output_size = 2):
        super().__init__()
        self.backbone = backbone
        self.output_size = output_size
        self.out_proj = nn.Dense(in_units=backbone.units,
                                 units=self.output_size,
                                 flatten=False)


    def forward(self, data, token_types, valid_length):
        _, pooled_out = self.backbone(data, token_types, valid_length)
        out = self.out_proj(pooled_out)
        return out

    def initialize_with_pretrained_backbone(self, backbone_params_path, ctx=None):
        self.backbone.load_parameters(backbone_params_path, ctx=ctx)
        self.out_proj.initialize(ctx=ctx)

