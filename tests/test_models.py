import tempfile
import pytest
from gluonnlp.models import get_backbone, list_backbone_names
from gluonnlp.utils.misc import count_parameters


def test_list_backbone_names():
    assert len(list_backbone_names()) > 0


@pytest.mark.parametrize('name', list_backbone_names())
def test_get_backbone(name):
    with tempfile.TemporaryDirectory() as root:
        model_cls, cfg, tokenizer, local_params_path, _ = get_backbone(name, root=root)
        net = model_cls.from_cfg(cfg)
        net.load_parameters(local_params_path)
        num_params, num_fixed_params = count_parameters(net.collect_params())
        assert num_params > 0
