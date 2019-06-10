import mxnet as mx
import numpy as np
import numpy.testing as npt
import io
from model import load_pretrained_GPT2
from transforms import GPT2Tokenizer, GPT2Detokenizer
from gluonnlp.vocab import Vocab

def test_pretrained_gpt2(ctx=None):
    sentence = ' natural language processing tools such as gluonnlp and torchtext'
    for model_name in ['117M', '345M']:
        if model_name == '117M':
            gt_logits = np.load('117M_gt_logits.npy')
        elif model_name == '345M':
            gt_logits = np.load('345M_gt_logits.npy')
        else:
            raise NotImplementedError
        model, vocab, tokenizer, detokenizer = load_pretrained_GPT2(model_name=model_name, ctx=ctx)
        model.hybridize()
        indices = vocab[tokenizer(sentence)]
        nd_indices = mx.nd.expand_dims(mx.nd.array(indices, ctx=ctx), axis=0)
        logits, new_states = model(nd_indices, None)
        npt.assert_allclose(logits.asnumpy(), gt_logits, 1E-5, 1E-5)

test_pretrained_gpt2(ctx=mx.gpu())

