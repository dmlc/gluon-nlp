import tempfile
import pytest
import mxnet as mx
import os
import numpy as np
import numpy.testing as npt
from gluonnlp.models import get_backbone, list_backbone_names
from gluonnlp.utils.parameter import count_parameters
from gluonnlp.utils.lazy_imports import try_import_tvm



def test_list_backbone_names():
    assert len(list_backbone_names()) > 0


def tvm_enabled():
    try:
        tvm = try_import_tvm()
        return True
    except:
        return False


@pytest.mark.slow
@pytest.mark.parametrize('name', list_backbone_names())
def test_get_backbone(name, device):
    with tempfile.TemporaryDirectory() as root, device:
        # Test for model download
        if name in ['google_t5_11B', 'google_mt5_xxl']: 
            pytest.skip('Skipping larger T5 (mT5) model test')
        model_cls, cfg, tokenizer, local_params_path, _ = get_backbone(name, root=root)
        net = model_cls.from_cfg(cfg)
        net.load_parameters(local_params_path)
        net.hybridize()
        num_params, num_fixed_params = count_parameters(net.collect_params())
        assert num_params > 0

        # Test for model export + save
        if 'gpt2' in name:
            pytest.skip('Skipping GPT-2 test')
        elif name in ['google_t5_3B', 'google_t5_11B', 'google_mt5_xl', 'google_mt5_xxl']: 
            pytest.skip('Skipping large T5 (mT5) model test')
        batch_size = 1
        sequence_length = 4
        inputs = mx.np.random.randint(0, 10, (batch_size, sequence_length))
        token_types = mx.np.random.randint(0, 2, (batch_size, sequence_length))
        valid_length = mx.np.random.randint(1, sequence_length, (batch_size,))
        if 'roberta' in name:
            out = net(inputs, valid_length)
        elif 'xlmr' in name:
            out = net(inputs, valid_length)
        elif 'bart' in name:
            out = net(inputs, valid_length, inputs, valid_length)
        elif 'gpt2' in name:
            states = net.init_states(batch_size=batch_size, device=device)
            out, new_states = net(inputs, states)
            out_np = out.asnumpy()
        elif 't5' in name:
            out = net(inputs, valid_length, inputs, valid_length)
        else:
            out = net(inputs, token_types, valid_length)
        mx.npx.waitall()
        net.export(os.path.join(root, 'model'))


@pytest.mark.serial
@pytest.mark.seed(123)
@pytest.mark.parametrize('model_name',
                         ['google_albert_base_v2',
                          'google_en_cased_bert_base',
                          'google_electra_small',
                          'fairseq_bart_base'])
@pytest.mark.parametrize('batch_size,seq_length', [(2, 4), (1, 4)])
@pytest.mark.parametrize('layout', ['NT', 'TN'])
@pytest.mark.skipif(not tvm_enabled(),
                    reason='TVM is not supported. So this test is skipped.')
def test_tvm_integration(model_name, batch_size, seq_length, layout, device):
    tvm = try_import_tvm()
    from tvm import relay
    from tvm.contrib import graph_executor
    from gluonnlp.utils.tvm_utils import get_ec2_tvm_flags, update_tvm_convert_map
    update_tvm_convert_map()
    tvm_recommended_flags = get_ec2_tvm_flags()
    if device.device_type == 'gpu':
        flags = tvm_recommended_flags['g4']
    elif device.device_type == 'cpu':
        flags = tvm_recommended_flags['c4']
        if model_name != 'google_albert_base_v2':
            # Skip all other tests
            return
    else:
        raise NotImplementedError
    with tempfile.TemporaryDirectory() as root, device:
        model_cls, cfg, tokenizer, backbone_param_path, _ = get_backbone(model_name, root=root)
        cfg.defrost()
        cfg.MODEL.layout = layout
        cfg.freeze()
        model = model_cls.from_cfg(cfg)
        model.load_parameters(backbone_param_path)
        model.hybridize()
        if layout == 'NT':
            token_ids = mx.np.random.randint(0, cfg.MODEL.vocab_size, (batch_size, seq_length),
                                             dtype=np.int32)
            token_types = mx.np.random.randint(0, 2, (batch_size, seq_length), dtype=np.int32)
            valid_length = mx.np.random.randint(seq_length // 2, seq_length, (batch_size,),
                                                dtype=np.int32)
        else:
            token_ids = mx.np.random.randint(0, cfg.MODEL.vocab_size, (seq_length, batch_size),
                                             dtype=np.int32)
            token_types = mx.np.random.randint(0, 2, (seq_length, batch_size), dtype=np.int32)
            valid_length = mx.np.random.randint(seq_length // 2, seq_length, (batch_size,),
                                                dtype=np.int32)
        if 'bart' in model_name:
            mx_out = model(token_ids, valid_length, token_ids, valid_length)
            shape_dict = {
                'data0': token_ids.shape,
                'data1': valid_length.shape,
                'data2': token_ids.shape,
                'data3': valid_length.shape,
            }
            dtype_dict = {
                'data0': token_ids.dtype.name,
                'data1': valid_length.dtype.name,
                'data2': token_ids.dtype.name,
                'data3': valid_length.dtype.name,
            }
        elif 'roberta' in model_name or 'xlmr' in model_name:
            mx_out = model(token_ids, valid_length)
            shape_dict = {
                'data0': token_ids.shape,
                'data1': valid_length.shape,
            }
            dtype_dict = {
                'data0': token_ids.dtype.name,
                'data1': valid_length.dtype.name,
            }
        else:
            mx_out = model(token_ids, token_types, valid_length)
            shape_dict = {
                'data0': token_ids.shape,
                'data1': token_types.shape,
                'data2': valid_length.shape
            }
            dtype_dict = {
                'data0': token_ids.dtype.name,
                'data1': token_types.dtype.name,
                'data2': valid_length.dtype.name
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
            device = tvm.gpu()
        else:
            device = tvm.cpu()
        rt = graph_executor.GraphModule(lib["default"](device))
        if 'bart' in model_name:
            rt.set_input(data0=token_ids.asnumpy(), data1=valid_length.asnumpy(), data2=token_ids.asnumpy(), data3=valid_length.asnumpy())
        elif 'roberta' in model_name:
            rt.set_input(data0=token_ids.asnumpy(), data1=valid_length.asnumpy())
        else:
            rt.set_input(data0=token_ids.asnumpy(), data1=token_types.asnumpy(), data2=valid_length.asnumpy())
        rt.run()
        for i in range(rt.get_num_outputs()):
            out = rt.get_output(i)
            if rt.get_num_outputs() == 1:
                mx_out_gt = mx_out.asnumpy()
            else:
                mx_out_gt = mx_out[i].asnumpy()
            npt.assert_allclose(out.asnumpy(), mx_out_gt, rtol=1e-3, atol=1e-1)
