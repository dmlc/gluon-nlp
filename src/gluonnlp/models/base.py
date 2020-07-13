from abc import ABC
from mxnet.gluon import HybridBlock
__all__ = ['HybridBlockWithLayout']


class HybridBlockWithLayout(HybridBlock, ABC):
    """Base class for a HybridBlock with temporal layout flag"""
    def __init__(self, layout, **kwargs):
        super().__init__(**kwargs)
        assert layout in ['TN', 'NT'], 'Invalid layout received = {}. ' \
                                       'Only "TN" and "NT" are accepted!'.format(layout)
        self._layout = layout

    @property
    def layout(self) -> str:
        return self._layout
