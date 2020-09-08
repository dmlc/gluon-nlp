import tempfile
import pytest
import mxnet as mx
import os
from gluonnlp.models import get_backbone, list_backbone_names
from gluonnlp.utils.misc import count_parameters
mx.npx.set_np()


def test_list_backbone_names():
    assert len(list_backbone_names()) > 0


@pytest.mark.slow
@pytest.mark.parametrize('name', list_backbone_names())
def test_get_backbone(name, ctx):
    with tempfile.TemporaryDirectory() as root, ctx:
        model_cls, cfg, tokenizer, local_params_path, _ = get_backbone(name, root=root)
        net = model_cls.from_cfg(cfg)
        net.load_parameters(local_params_path)
        net.hybridize()
        num_params, num_fixed_params = count_parameters(net.collect_params())
        assert num_params > 0

        # Test for model export + save
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
            # Temporarily skip GPT-2 test
            return
        else:
            out = net(inputs, token_types, valid_length)
        mx.npx.waitall()
        net.export(os.path.join(root, 'model'))
