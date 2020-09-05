import numpy as np
import random
import os
import mxnet as mx
from mxnet import gluon
import argparse
import logging
import time
from gluonnlp.utils.misc import logging_config
from gluonnlp.models.transformer import TransformerModel,\
    TransformerNMTInference
from gluonnlp.data.batchify import Tuple, Pad, Stack
from gluonnlp.data.filtering import MosesNormalizer
from gluonnlp.data import tokenizers
from gluonnlp.sequence_sampler import BeamSearchSampler, BaseStepDecoder
import sacrebleu
from tqdm import tqdm

from gluonnlp.models.gpt2 import GPT2ForLM, list_pretrained_gpt2, get_pretrained_gpt2

mx.npx.set_np()


def parse_args():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--model_name', type=str, default='gpt2_124M',
                        choices=list_pretrained_gpt2(), help='')
    parser.add_argument('--seed', type=int, default=None, help='The random seed.')
    parser.add_argument('--nsamples', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--length', type=int, default=None, help='')
    parser.add_argument('--temperature', type=float, default=1.0, help='')
    parser.add_argument('--top_k', type=int, default=-1, help='')
    parser.add_argument('--top_p', type=float, required=-1.0, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')



# input = prev , states = None, output += new samples ()
# 输入start token时 一直
# 输入context 时


class GPT2Decoder(BaseStepDecoder):
    def __init__(self, gpt2_lm_model):
        self._gpt2_lm_model = gpt2_lm_model
    @property
    def state_batch_axis(self):
        return 2 if self._gpt2_lm_model._backbone_model.layout == 'NT' else 3
    def init_states(self, batch_size, ctx):
        return self._gpt2_lm_model.init_states(batch_size, ctx)
    def __call__(self, data, states):
        return self._gpt2_lm_model(data, states)


def sample_gpt2(args):
    ctx = mx.gpu(args.gpu) if args.gpu is not None else \
          mx.cpu()
    
    cfg, tokenizer, _, lm_params_path = get_pretrained_gpt2(
        model_name=args.model_name,
        load_backbone=False,
        load_lm=True)
    
    if args.length is None:
        args.length = cfg.MODEL.max_length
    assert args.length <= cfg.MODEL.max_length, \
           "Can't get samples longer than window size: {}".format(cfg.MODEL.max_length)
    
    model = GPT2ForLM(cfg)
    model.hybridize()
    model.load_parameters(lm_params_path, ctx=ctx)
    gpt2decoder = GPT2Decoder(model)
    
    sampler = BeamSearchSampler(
        beam_size=1,
        decoder=gpt2decoder,
        eos_id=tokenizer.eos_id,
        vocab_size=cfg.MODEL.vocab_size,
        max_length_a=0,
        max_length_b=cfg.MODEL.max_length,
        min_length=1,
        temperature=args.temperature,
        sampling=True,
        sampling_topp=args.top_p,
        sampling_topk=args.top_k,
        early_return=True
    )
    
    start_input = mx.np.full((args.batch_size, 1), tokenizer.bos_id)
    start_states = gpt2decoder.init_states(args.batch_size, ctx)
    
    generated = 0
    while args.nsamples <= 0 or generated < args.nsamples:
        samples = sampler(start_input, start_states)
        for i in args.batch_size:
            text = tokenizer.decode(samples[i][0])
            print(text)
        generated += args.batch_size


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    args = parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    random.seed(args.seed)
    sample_gpt2(args)