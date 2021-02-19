# TVM Demo


```{.python .input}
import mxnet as mx
import numpy as np
from gluonnlp.models import get_backbone
from gluonnlp.utils.lazy_imports import try_import_tvm
from gluonnlp.data.batchify import Pad, Stack
mx.npx.set_np()
ctx = mx.gpu()
```

## Load the ELECTRA-base


```{.python .input}
import os
model_name = 'google_electra_base'
model_cls, cfg, tokenizer, backbone_param_path, _ = get_backbone(model_name)
model = model_cls.from_cfg(cfg)
model.hybridize()
model.load_parameters(backbone_param_path, ctx=ctx)
```


```{.python .input}
sentences = ['hello world', 'orbit workbench demo via gluon toolkits']
tokens = tokenizer.encode(sentences, int)
tokens = [[tokenizer.vocab.cls_id] + tokens[0] + [tokenizer.vocab.sep_id],
          [tokenizer.vocab.cls_id] + tokens[1] + [tokenizer.vocab.sep_id]]
print(tokens)
```


```{.python .input}
token_ids = Pad()(tokens)
valid_length = Stack()(list(map(len, tokens)))
segment_ids = np.zeros_like(token_ids)
print(token_ids)
print(valid_length)
```


```{.python .input}
contextual_embeddings, cls_embedding = model(mx.np.array(token_ids, ctx=ctx),
            mx.np.array(segment_ids, ctx=ctx), 
            mx.np.array(valid_length, ctx=ctx))
```


```{.python .input}
contextual_embeddings
```


```{.python .input}
cls_embedding
```

## Use TVM for Inference


```{.python .input}
_TVM_RT_CACHE = dict()


def compile_tvm_graph_runtime(model, model_name, cfg,
                              batch_size, seq_length, dtype, instance_type):
    layout = cfg.MODEL.layout
    compute_layout = cfg.MODEL.compute_layout
    key = (model_name, layout, compute_layout, batch_size, seq_length, dtype, instance_type)
    if key in _TVM_RT_CACHE:
        return _TVM_RT_CACHE[key]
    tvm = try_import_tvm()
    from tvm import relay
    from tvm.contrib import graph_runtime
    from gluonnlp.utils.tvm_utils import get_ec2_tvm_flags, update_tvm_convert_map
    flags = get_ec2_tvm_flags()[instance_type]
    update_tvm_convert_map()
    token_ids_shape = (batch_size, seq_length) if layout == 'NT' else (seq_length, batch_size)
    valid_length_shape = (batch_size,)
    if 'bart' in model_name:
        shape_dict = {
            'data0': token_ids_shape,
            'data1': valid_length_shape,
            'data2': token_ids_shape,
            'data3': valid_length_shape,
        }
        dtype_dict = {
            'data0': 'int32',
            'data1': 'int32',
            'data2': 'int32',
            'data3': 'int32',
        }
    elif 'roberta' in model_name or 'xlmr' in model_name:
        shape_dict = {
            'data0': token_ids_shape,
            'data1': valid_length_shape,
        }
        dtype_dict = {
            'data0': 'int32',
            'data1': 'int32',
        }
    else:
        shape_dict = {
            'data0': token_ids_shape,
            'data1': token_ids_shape,
            'data2': valid_length_shape,
        }
        dtype_dict = {
            'data0': 'int32',
            'data1': 'int32',
            'data2': 'int32'
        }
    sym = model._cached_graph[1]
    params = {}
    for k, v in model.collect_params().items():
        params[v._var_name] = tvm.nd.array(v.data().asnumpy())
    mod, params = relay.frontend.from_mxnet(sym, shape=shape_dict, dtype=dtype_dict, arg_params=params)
    target = flags['target']
    use_gpu = flags['use_gpu']
    opt_level = flags['opt_level']
    required_pass = flags['required_pass']
    with tvm.transform.PassContext(opt_level=opt_level, required_pass=required_pass):
        lib = relay.build(mod, target, params=params)
    if use_gpu:
        ctx = tvm.gpu()
    else:
        ctx = tvm.cpu()
    rt = graph_runtime.GraphModule(lib["default"](ctx))
    _TVM_RT_CACHE[key] = rt
    return rt
```


```{.python .input}
rt = compile_tvm_graph_runtime(model, model_name, cfg, token_ids.shape[0],
                               token_ids.shape[1], 'float32', 'g4')
```


```{.python .input}
rt.set_input(data0=token_ids.asnumpy(), data1=segment_ids.asnumpy(), data2=valid_length.asnumpy())
rt.run()
tvm_contextual_embedding = rt.get_output(0)
tvm_cls_embedding = rt.get_output(1)
```


```{.python .input}
tvm_cls_embedding
```
