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
"""BERT datasets."""

__all__ = [
    'MRPCDataset', 'QQPDataset', 'QNLIDataset', 'RTEDataset', 'STSBDataset',
    'COLADataset', 'MNLIDataset', 'WNLIDataset', 'SSTDataset', 'BertEmbeddingDataset',
    'BERTDatasetTransform'
]

import os
import warnings
import numpy as np
from mxnet.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric
from mxnet.gluon.data import Dataset
from gluonnlp.data import TSVDataset, BERTSentenceTransform
from gluonnlp.data.registry import register


@register(segment=['train', 'dev', 'test'])
class MRPCDataset(TSVDataset):
    """The Microsoft Research Paraphrase Corpus dataset.
    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/MRPC'
        Path to the folder which stores the MRPC dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """
    task_name = 'MRPC'
    is_pair = True

    def __init__(self,
                 segment='train',
                 root=os.path.join(
                     os.getenv('GLUE_DIR', 'glue_data'), task_name)):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        A_IDX, B_IDX, LABEL_IDX = 3, 4, 0
        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(MRPCDataset, self).__init__(
            path, num_discard_samples=1, field_indices=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy and F1"""
        metric = CompositeEvalMetric()
        for child_metric in [Accuracy(), F1()]:
            metric.add(child_metric)
        return metric


class GLUEDataset(TSVDataset):
    """GLUEDataset class"""

    def __init__(self, path, num_discard_samples, fields):
        self.fields = fields
        super(GLUEDataset, self).__init__(
            path, num_discard_samples=num_discard_samples)

    def _read(self):
        all_samples = super(GLUEDataset, self)._read()
        largest_field = max(self.fields)
        # to filter out error records
        final_samples = [[s[f] for f in self.fields] for s in all_samples
                         if len(s) >= largest_field + 1]
        residuals = len(all_samples) - len(final_samples)
        if residuals > 0:
            warnings.warn(
                '{} samples have been filtered out due to parsing error.'.
                format(residuals))
        return final_samples


@register(segment=['train', 'dev', 'test'])
class QQPDataset(GLUEDataset):
    """Dataset for Quora Question Pairs.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/QQP'
        Path to the folder which stores the QQP dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """
    task_name = 'QQP'
    is_pair = True

    def __init__(self,
                 segment='train',
                 root=os.path.join(
                     os.getenv('GLUE_DIR', 'glue_data'), task_name)):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 3, 4, 5
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX, B_IDX = 1, 2
            fields = [A_IDX, B_IDX]
        super(QQPDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy and F1"""
        metric = CompositeEvalMetric()
        for child_metric in [Accuracy(), F1()]:
            metric.add(child_metric)
        return metric


@register(segment=['train', 'dev', 'test'])
class RTEDataset(GLUEDataset):
    """Task class for Recognizing Textual Entailment

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/RTE'
        Path to the folder which stores the RTE dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """
    task_name = 'RTE'
    is_pair = True

    def __init__(self,
                 segment='train',
                 root=os.path.join(
                     os.getenv('GLUE_DIR', 'glue_data'), task_name)):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 1, 2, 3
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX, B_IDX = 1, 2
            fields = [A_IDX, B_IDX]
        super(RTEDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['not_entailment', 'entailment']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()


@register(segment=['train', 'dev', 'test'])
class QNLIDataset(GLUEDataset):
    """Task class for SQuAD NLI

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/QNLI'
        Path to the folder which stores the QNLI dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """
    task_name = 'QNLI'
    is_pair = True

    def __init__(self,
                 segment='train',
                 root=os.path.join(
                     os.getenv('GLUE_DIR', 'glue_data'), task_name)):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 1, 2, 3
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX, B_IDX = 1, 2
            fields = [A_IDX, B_IDX]
        super(QNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['not_entailment', 'entailment']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()


@register(segment=['train', 'dev', 'test'])
class STSBDataset(GLUEDataset):
    """Task class for Sentence Textual Similarity Benchmark.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/STS-B'
        Path to the folder which stores the STS dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """
    task_name = 'STS-B'
    is_pair = True

    def __init__(self,
                 segment='train',
                 root=os.path.join(
                     os.getenv('GLUE_DIR', 'glue_data'), task_name)):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 7, 8, 9
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX, B_IDX = 7, 8
            fields = [A_IDX, B_IDX]
        super(STSBDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_metric():
        """
        Get metrics Accuracy
        """
        return PearsonCorrelation()

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return None


@register(segment=['train', 'dev', 'test'])
class COLADataset(GLUEDataset):
    """Class for Warstdadt acceptability task

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/CoLA
        Path to the folder which stores the CoLA dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """
    task_name = 'CoLA'
    is_pair = False

    def __init__(self,
                 segment='train',
                 root=os.path.join(
                     os.getenv('GLUE_DIR', 'glue_data'), task_name)):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, LABEL_IDX = 3, 1
            fields = [A_IDX, LABEL_IDX]
            super(COLADataset, self).__init__(
                path, num_discard_samples=0, fields=fields)
        elif segment == 'test':
            A_IDX = 3
            fields = [A_IDX]
            super(COLADataset, self).__init__(
                path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_metric():
        """Get metrics  Matthews Correlation Coefficient"""
        return MCC(average='micro')

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']


@register(segment=['train', 'dev', 'test'])
class SSTDataset(GLUEDataset):
    """Task class for Stanford Sentiment Treebank.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/SST-2
        Path to the folder which stores the SST-2 dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """
    task_name = 'SST'
    is_pair = False

    def __init__(self,
                 segment='train',
                 root=os.path.join(
                     os.getenv('GLUE_DIR', 'glue_data'), 'SST-2')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, LABEL_IDX = 0, 1
            fields = [A_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX = 1
            fields = [A_IDX]
        super(SSTDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']


@register(segment=[
    'dev_matched', 'dev_mismatched', 'test_matched', 'test_mismatched',
    'train'
])  # pylint: disable=c0301
class MNLIDataset(GLUEDataset):
    """Task class for Multi-Genre Natural Language Inference
    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'dev_matched', 'dev_mismatched',
        'test_matched', 'test_mismatched', 'train' or their combinations.
    root : str, default '$GLUE_DIR/MNLI'
        Path to the folder which stores the MNLI dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    task_name = 'MNLI'
    is_pair = True

    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'MNLI')):  # pylint: disable=c0330
        self._supported_segments = [
            'train', 'dev_matched', 'dev_mismatched',
            'test_matched', 'test_mismatched',
        ]
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev_matched', 'dev_mismatched']:
            A_IDX, B_IDX = 8, 9
            LABEL_IDX = 11 if segment == 'train' else 15
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment in ['test_matched', 'test_mismatched']:
            A_IDX, B_IDX = 8, 9
            fields = [A_IDX, B_IDX]
        super(MNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['neutral', 'entailment', 'contradiction']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()


@register(segment=['train', 'dev', 'test'])
class WNLIDataset(GLUEDataset):
    """Class for Winograd NLI task

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test' or their combinations.
    root : str, default '$GLUE_DIR/WNLI'
        Path to the folder which stores the WNLI dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """
    task_name = 'WNLI'
    is_pair = True

    def __init__(self,
                 segment='train',
                 root=os.path.join(
                     os.getenv('GLUE_DIR', 'glue_data'), task_name)):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 1, 2, 3
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX, B_IDX = 1, 2
            fields = [A_IDX, B_IDX]
        super(WNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()


class BERTDatasetTransform(object):
    """Dataset Transformation for BERT-style Sentence Classification or Regression.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    labels : list of int , float or None. defaults None
        List of all label ids for the classification task and regressing task.
        If labels is None, the default task is regression
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    label_dtype: int32 or float32, default float32
        label_dtype = int32 for classification task
        label_dtype = float32 for regression task
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 labels=None,
                 pad=True,
                 pair=True,
                 label_dtype='float32'):
        self.label_dtype = label_dtype
        self.labels = labels
        if self.labels:
            self._label_map = {}
            for (i, label) in enumerate(labels):
                self._label_map[label] = i
        self._bert_xform = BERTSentenceTransform(
            tokenizer, max_seq_length, pad=pad, pair=pair)

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 3 strings:
        text_a, text_b and label.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
            label: '0'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
            label: 0

        For single sequences, the input is a tuple of 2 strings: text_a and label.
        Inputs:
            text_a: 'the dog is hairy .'
            label: '1'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
            label: 1

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b, label). For single sequences, the input is a tuple
            of 2 strings: (text_a, label).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        np.array: classification task: label id in 'int32', shape (batch_size, 1),
            regression task: label in 'float32', shape (batch_size, 1)
        """
        input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])

        label = line[-1]
        if self.labels:  # for classification task
            label = self._label_map[label]
        label = np.array([label], dtype=self.label_dtype)

        return input_ids, valid_length, segment_ids, label


class BertEmbeddingDataset(Dataset):
    """Dataset for BERT Embedding

    Parameters
    ----------
    sentences : List[str].
        Sentences for embeddings.
    transform : BERTDatasetTransform, default None.
        transformer for BERT input format
    """

    def __init__(self, sentences, transform=None):
        """Dataset for BERT Embedding

        Parameters
        ----------
        sentences : List[str].
            Sentences for embeddings.
    transform : BERTDatasetTransform, default None.
            transformer for BERT input format
        """
        self.sentences = sentences
        self.transform = transform

    def __getitem__(self, idx):
        sentence = (self.sentences[idx], 0)
        if self.transform:
            return self.transform(sentence)
        else:
            return sentence

    def __len__(self):
        return len(self.sentences)
