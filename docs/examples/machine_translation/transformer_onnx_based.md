# Running MXNet-trained Transformer with ONNXRuntime

In [Using Pre-trained Transformer](https://nlp.gluon.ai/examples/machine_translation/transformer.html) we have seen how to run a pretrained MXNet transformer model for end-2-end machine translation. In this blog, we are going to export the transformer model to the ONNX format, run inference with ONNXRuntime, and achieve the same end-to-end translation as before.

## Setup

```{.python .input}
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import mxnet as mx
import gluonnlp as nlp
# make sure gluonnlp version is >= 0.7.0
nlp.utils.check_version('0.7.0')

# use cpu context to load the model
ctx = mx.cpu(0)
print('ctx: ', ctx)
```

## Load the Pre-trained Transformer 

```{.python .input}
# load the model
wmt_model_name = 'transformer_en_de_512'
wmt_transformer_model, wmt_src_vocab, wmt_tgt_vocab = \
    nlp.model.get_model(wmt_model_name,
                        dataset_name='WMT2014',
                        pretrained=True,
                        ctx=ctx)

# we are using mixed vocab of EN-DE, so the source and target language vocab are the same
print('EN size: ', len(wmt_src_vocab), '\nDE size: ', len(wmt_tgt_vocab))
```

## Save the Components of the Transformer as MXNet Models

Note that the Transformer, which is an instance of class `NMTModel`, is not a monolith, but a collection of several smaller components such as `src_embed`, `encoder`, `tgt_embed`, `one_step_ahead_decoder`, and `tgt_proj`. Those components are by themselves MXNet hybrid models. In `NMTModel`, there are high-level member functions such as `encode` and `decode_step`, and they will in turn call the finer components in combination. For example, `encode` internally uses both `src_embed` and `encoder`. In the cell below we are going to create some dummy input data and call `encode` and`decode_step`. This is to make sure that we run foward path on all the hybridized components at least once. Then, we can [save the architecture and the parameters](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html) of those components.

```{.python .input}
# save the necessary components of transformer
import os

wmt_transformer_model.hybridize(static_alloc=False)

# the transformer model consists of several components, among which we need the following to run inference
print(type(wmt_transformer_model.src_embed))
print(type(wmt_transformer_model.encoder))
print(type(wmt_transformer_model.tgt_embed))
print(type(wmt_transformer_model.one_step_ahead_decoder))
print(type(wmt_transformer_model.tgt_proj))

# define some dummy data
batch = 1
seq_length = 16
C_in = 512
C_out = 512
src = mx.nd.random.uniform(0, 36794, shape=(batch, seq_length), dtype='float32')
step_input = mx.nd.random.uniform(0, 36794, shape=(batch,), dtype='float32')
src_valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

# run forward once with the following functions so taht we can export the components
## encode() internally calls src_embed and encoder
encoder_outputs, _ = wmt_transformer_model.encode(src, valid_length=src_valid_length)
## init_state_from_encoder() helps prepare decoder_states
decoder_states = wmt_transformer_model.decoder.init_state_from_encoder(encoder_outputs,
                                                                       src_valid_length)
## decode_step() internally calls tgt_embed, one_step_ahead_decoder, tgt_proj
_, _, _ = wmt_transformer_model.decode_step(step_input, decoder_states)

# export the components
base_path = './components'
if not os.path.exists(base_path):
    os.makedirs(base_path)
for component in ['src_embed', 'encoder', 'tgt_embed', 'one_step_ahead_decoder', 'tgt_proj']:
    prefix = "%s/%s" %(base_path, component)
    component = getattr(wmt_transformer_model, component)
    component.export(prefix)
    sym_file = "%s-symbol.json" % prefix
    params_file = "%s-0000.params" % prefix

print('Files under ./components \n', os.listdir(base_path))
```

## Export the MXNet Transformer Components to the ONNX Format

Now that we have saved the necessary components of the transformer model, we are going to export each of them to the ONNX format. Here, notice that we are using the dynamic input feature of mx2onnx as the input batch and sequence lenghth can vary depending on different input sentences or beam search widths.

Please note that MXNet version 1.9 or above is required for this step.

```{.python .input}
# export the transformer components to ONNX models

from mxnet import onnx as mx2onnx

def export_to_onnx(prefix, input_shapes, input_types, **kwargs):
    sym_file = "%s-symbol.json" % prefix
    params_file = "%s-0000.params" % prefix
    onnx_file = "%s.onnx" % prefix
    return mx2onnx.export_model(sym_file, params_file, input_shapes, input_types,
                                onnx_file, **kwargs)

# export src_embed
prefix = "%s/src_embed" %base_path
input_shapes = [(batch, seq_length)]
dynamic_input_shapes = [(batch, 'seq_length')]
input_types = [np.float32]
onnx_file = export_to_onnx(prefix, input_shapes, input_types, dynamic=True,
                           dynamic_input_shapes=dynamic_input_shapes)

# export encoder
prefix = "%s/encoder" %base_path
input_shapes = [(batch, seq_length, C_in), (batch,)]
dynamic_input_shapes = [(batch, 'seq_length', C_in), (batch,)]
input_types = [np.float32, np.float32]
onnx_file = export_to_onnx(prefix, input_shapes, input_types, dynamic=True,
            dynamic_input_shapes=dynamic_input_shapes)

# export tgt_embed
prefix = "%s/tgt_embed" %base_path
input_shapes = [(batch,)]
dynamic_input_shapes = [('batch',)]
input_types = [np.int32]
onnx_file = export_to_onnx(prefix, input_shapes, input_types, dynamic=True,
                           dynamic_input_shapes=dynamic_input_shapes)

# export one_step_ahead_decoder
prefix = "%s/one_step_ahead_decoder" %base_path
# mem_masks, decoder_inputs, mem_value
input_shapes = [(batch, seq_length), (batch, 1, C_in), (batch, seq_length, C_out)]
dynamic_input_shapes = [('batch', 'seq_length'), ('batch', 'cur_step_seq_length', C_in),
                        ('batch', 'seq_length', C_out)]
input_types = [np.float32, np.float32, np.float32]
onnx_file = export_to_onnx(prefix, input_shapes, input_types, dynamic=True,
                           dynamic_input_shapes=dynamic_input_shapes)

# export tgt_proj
prefix = "%s/tgt_proj" %base_path
input_shapes = [(batch,  C_out)]
dynamic_input_shapes = [('batch', C_out)]
input_types = [np.float32]
onnx_file = export_to_onnx(prefix, input_shapes, input_types, dynamic=True,
                           dynamic_input_shapes=dynamic_input_shapes)

print('Files under ./components \n', os.listdir(base_path))
```

## Beam Search Hyper-Parameters

Now that we have the exported transformer components (in the ONNX format), we can run them with any runtime framework that supports ONNX models and use a custom beam search implementation. However, in this tutorial we are going to stick to the same beam seach as in the origina GluonNLP [transformer tutorial](https://nlp.gluon.ai/examples/machine_translation/transformer.html). Let's review the hyper parameters before we proceed.

```{.python .input}
import hyperparameters as hparams
# check the hyper-parameters
print('beam_size:'.ljust(12), hparams.beam_size)
print('lp_alpha:'.ljust(12), hparams.lp_alpha)
print('lp_k:'.ljust(12), hparams.lp_k)
```

## Define a Translator with a Custom NMTModel

in the original [transformer tutorial](https://nlp.gluon.ai/examples/machine_translation/transformer.html) we use a `BeamSearchTranslator` to run end-to-end machine translation task. `BeamSearchTranslator` would take in a `NMTModel` which our pre-trained transformer model is an instance of, and use it to make predictions word by word. 

```python
# in class BeamSearchTranslator
class BeamSearchTranslator:
    ......
    def _decode_logprob(self, step_input, states):
        out, states, _ = self._model.decode_step(step_input, states)
        return mx.nd.log_softmax(out), states

    def translate(self, src_seq, src_valid_length):
        batch_size = src_seq.shape[0]
        encoder_outputs, _ = self._model.encode(src_seq, valid_length=src_valid_length)
        decoder_states = self._model.decoder.init_state_from_encoder(encoder_outputs,
                                                                     src_valid_length)
        inputs = mx.nd.full(shape=(batch_size,), ctx=src_seq.context, dtype=np.float32,
                            val=self._model.tgt_vocab.token_to_idx[
                                self._model.tgt_vocab.bos_token])
        samples, scores, sample_valid_length = self._sampler(inputs, decoder_states)
        return samples, scores, sample_valid_length
```

Here we can see that a `BeamSearchTranslator` will make calls to `encode`, `decode_step`, `decoder.init_state_from_encoder`, `tgt_vocab.token_to_idx`, which are functions or objects defined in `NMTModel`. This means, if we can define a customized `NMTModel` class, say `ONNXNMTModel`, and define those same interfaces, then it would be compatible with `BeamSearchTranslator`. Whether within this customized `ONNXNMTModel` we call the original MXNet model or use the exported ONNX models with ONNXRuntime, it would not matter from the `BeamSearchTranslator`'s perspecrive. If you are intrested in this customized class, you can refer to `CustomNMTModel.py` for the full implementation.

```{.python .input}
import nmt
import utils
import CustomNMTModel

# detokenizer
wmt_detokenizer = nlp.data.SacreMosesDetokenizer()

# create a custom ONNXNMTModel
onnxnmtmodel = CustomNMTModel.ONNXNMTModel(wmt_transformer_model.tgt_vocab,
                                           'components/src_embed.onnx',
                                           'components/encoder.onnx',
                                           'components/tgt_embed.onnx',
                                           'components/one_step_ahead_decoder.onnx',
                                           'components/tgt_proj.onnx')

# define beam search translator
onnx_wmt_translator = nmt.translation.BeamSearchTranslator(
    # note here that we are using an ONNXNMTModel object to replace the actual
    # transformer model which is an NMTModel object 
    model=onnxnmtmodel, # wmt_transformer_model,
    beam_size=hparams.beam_size,
    scorer=nlp.model.BeamSearchScorer(alpha=hparams.lp_alpha, K=hparams.lp_k),
    max_length=200)


print(type(wmt_transformer_model))
print(type(onnxnmtmodel))
```

## Machine Translation powered by MXNet+GluonNLP... Plus MX2ONNX+ONNXRuntime

Now that we have a `BeamSearchTranslator` defined, we can run end-to-end inference on it! 

```{.python .input}
# input English sentence
print('Translate the following English sentence into German:')
sample_src_seq = 'I am a software engineer and I love to play football .'
print('[\'' + sample_src_seq + '\']')

# run end2end inference using onnxruntime + gluonnlp beam search
sample_tgt_seq = utils.translate(onnx_wmt_translator,
                                 sample_src_seq,
                                 wmt_src_vocab,
                                 wmt_tgt_vocab,
                                 wmt_detokenizer,
                                 ctx)

# output German sentence
print('The German translation is:')
print(sample_tgt_seq)
```
