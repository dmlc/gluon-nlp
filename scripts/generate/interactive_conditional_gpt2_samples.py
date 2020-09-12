import os
import mxnet as mx
import argparse
from gluonnlp.utils import set_seed
from gluonnlp.sequence_sampler import BeamSearchSampler, BaseStepDecoder
from gluonnlp.models.gpt2 import GPT2ForLM, list_pretrained_gpt2, get_pretrained_gpt2

mx.npx.set_np()

def parse_args():
    parser = argparse.ArgumentParser(
        description='GPT-2 unconditional sampler. Load a GPT-2 model and sample.')
    parser.add_argument('--model_name', type=str, default='gpt2_124M',
                        choices=list_pretrained_gpt2(), help='Model name')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--nsamples', type=int, default=0, help='Number of samples to return')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batches')
    parser.add_argument('--length', type=int, default=None,
                        help='Number of tokens in generated text, if None (default), is '
                             'determined by model max_length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='')
    parser.add_argument('--top_k', type=int, default=-1,
                        help='Multinomial sampling with topk, '
                            'see [ACL2018] "Hierarchical Neural Story Generation"'
                            'https://www.aclweb.org/anthology/P18-1082.pdf')
    parser.add_argument('--top_p', type=float, default=-1.0,
                        help='Multinomial sampling with topp, '
                             'see [ICLR2020] "The Curious Case of Neural Text Degeneration"'
                             'https://arxiv.org/abs/1904.09751')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Which gpu to use, set None to use cpu')
    return parser.parse_args()


class GPT2Decoder(BaseStepDecoder):
    def __init__(self, gpt2_lm_model):
        self._gpt2_lm_model = gpt2_lm_model
    @property
    def state_batch_axis(self):
        return 2 if self._gpt2_lm_model._backbone_model.layout == 'NT' else 3
    def init_states(self, batch_size, ctx):
        return self._gpt2_lm_model.init_states(batch_size, ctx)
    def __call__(self, data, states):
        data = mx.npx.reshape(data, (-2, -1))
        logits, new_states = self._gpt2_lm_model(data, states)
        return logits[:,-1,:], new_states


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
        eos_id=None,
        vocab_size=cfg.MODEL.vocab_size,
        max_length_a=0,
        max_length_b=args.length,
        min_length=1,
        temperature=args.temperature,
        sampling=True,
        sampling_topp=args.top_p,
        sampling_topk=args.top_k,
        early_return=False
    )
    start_states = gpt2decoder.init_states(args.batch_size, ctx)
    
    while True:
        raw_text = input('Model prompt >>> ')
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("Model prompt >>> ")
        context_tokens = tokenizer.encode(raw_text, output_type=int)
        start_input = mx.np.repeat(mx.np.expand_dims(mx.np.array(context_tokens, ctx=ctx), 0),
                                   args.batch_size,
                                   axis=0)
        generated = 0
        while generated < args.nsamples:
            samples, _, _ = sampler(start_input, start_states)
            for i in range(args.batch_size):
                generated += 1
                ids = samples[i][0].asnumpy().tolist()
                ids = ids[1:ids.index(-1)] if -1 in ids else \
                      ids[1:]
                text = tokenizer.decode(ids)
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
        print("=" * 80)

if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    sample_gpt2(args)
