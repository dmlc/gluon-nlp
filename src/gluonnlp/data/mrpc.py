# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MRPCDatasets."""

__all__ = ['MRPCDataset']

import os

from . import TSVDataset
from .registry import register


@register(segment=['train', 'dev', 'test'])
class MRPCDataset(TSVDataset):
    """The Microsoft Research Paraphrase Corpus dataset.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test' or their combinations.
    root : str, default '$GLUE_DIR/MRPC'
        Path to the folder which stores the MRPC dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self, segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'), 'MRPC')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        A_IDX, B_IDX, LABEL_IDX = 3, 4, 0
        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(MRPCDataset, self).__init__(path, num_discard_samples=1, field_indices=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']
