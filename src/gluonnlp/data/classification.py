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
"""GLUE classification/regression datasets."""


__all__ = [
    'MRPCTask', 'QQPTask', 'QNLITask', 'RTETask', 'STSBTask',
    'CoLATask', 'MNLITask', 'WNLITask', 'SSTTask', 'XNLITask', 'get_task'
]

from copy import copy
from mxnet.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric
from .glue import GlueCoLA, GlueSST2, GlueSTSB, GlueMRPC
from .glue import GlueQQP, GlueRTE, GlueMNLI, GlueQNLI, GlueWNLI
from .baidu_ernie_data import BaiduErnieXNLI, BaiduErnieChnSentiCorp, BaiduErnieLCQMC


class GlueTask:
    """Abstract GLUE task class.

    Parameters
    ----------
    class_labels : list of str, or None
        Classification labels of the task.
        Set to None for regression tasks with continuous real values.
    metrics : list of EValMetric
        Evaluation metrics of the task.
    is_pair : bool
        Whether the task deals with sentence pairs or single sentences.
    label_alias : dict
        label alias dict, some different labels in dataset actually means
        the same. e.g.: {'contradictory':'contradiction'} means contradictory
        and contradiction label means the same in dataset, they will get
        the same class id.
    """
    def __init__(self, class_labels, metrics, is_pair, label_alias=None):
        self.class_labels = class_labels
        self.metrics = metrics
        self.is_pair = is_pair
        self.label_alias = label_alias

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for the task.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments.

        Returns
        -------
        TSVDataset : the dataset of target segment.
        """
        raise NotImplementedError()

    def dataset_train(self):
        """Get the training segment of the dataset for the task.

        Returns
        -------
        tuple of str, TSVDataset : the segment name, and the dataset.
        """
        return 'train', self.get_dataset(segment='train')

    def dataset_dev(self):
        """Get the dev segment of the dataset for the task.

        Returns
        -------
        tuple of (str, TSVDataset), or list of tuple : the segment name, and the dataset.
        """
        return 'dev', self.get_dataset(segment='dev')

    def dataset_test(self):
        """Get the test segment of the dataset for the task.

        Returns
        -------
        tuple of (str, TSVDataset), or list of tuple : the segment name, and the dataset.
        """
        return 'test', self.get_dataset(segment='test')

class MRPCTask(GlueTask):
    """The MRPC task on GlueBenchmark.

    Examples
    --------
    >>> MRPC = MRPCTask()
    >>> MRPC.class_labels
    ['0', '1']
    >>> type(MRPC.metrics.get_metric(0))
    <class 'mxnet.metric.Accuracy'>
    >>> type(MRPC.metrics.get_metric(1))
    <class 'mxnet.metric.F1'>
    >>> MRPC.dataset_train()[0]
    -etc-
    'train'
    >>> len(MRPC.dataset_train()[1])
    3668
    >>> MRPC.dataset_dev()[0]
    'dev'
    >>> len(MRPC.dataset_dev()[1])
    408
    >>> MRPC.dataset_test()[0]
    -etc-
    'test'
    >>> len(MRPC.dataset_test()[1])
    1725
    """
    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = CompositeEvalMetric()
        metric.add(Accuracy())
        metric.add(F1(average='micro'))
        super(MRPCTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for MRPC.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueMRPC(segment=segment)

class QQPTask(GlueTask):
    """The Quora Question Pairs task on GlueBenchmark.

    Examples
    --------
    >>> QQP = QQPTask()
    >>> QQP.class_labels
    ['0', '1']
    >>> type(QQP.metrics.get_metric(0))
    <class 'mxnet.metric.Accuracy'>
    >>> type(QQP.metrics.get_metric(1))
    <class 'mxnet.metric.F1'>
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     # Ignore warnings triggered by invalid entries in GlueQQP set
    ...     warnings.simplefilter("ignore")
    ...     QQP.dataset_train()[0]
    -etc-
    'train'
    >>> QQP.dataset_test()[0]
    -etc-
    'test'
    >>> len(QQP.dataset_test()[1])
    390965
    """
    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = CompositeEvalMetric()
        metric.add(Accuracy())
        metric.add(F1(average='micro'))
        super(QQPTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for QQP.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueQQP(segment=segment)


class RTETask(GlueTask):
    """The Recognizing Textual Entailment task on GlueBenchmark.

    Examples
    --------
    >>> RTE = RTETask()
    >>> RTE.class_labels
    ['not_entailment', 'entailment']
    >>> type(RTE.metrics)
    <class 'mxnet.metric.Accuracy'>
    >>> RTE.dataset_train()[0]
    -etc-
    'train'
    >>> len(RTE.dataset_train()[1])
    2490
    >>> RTE.dataset_dev()[0]
    -etc-
    'dev'
    >>> len(RTE.dataset_dev()[1])
    277
    >>> RTE.dataset_test()[0]
    -etc-
    'test'
    >>> len(RTE.dataset_test()[1])
    3000
    """
    def __init__(self):
        is_pair = True
        class_labels = ['not_entailment', 'entailment']
        metric = Accuracy()
        super(RTETask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for RTE.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueRTE(segment=segment)

class QNLITask(GlueTask):
    """The SQuAD NLI task on GlueBenchmark.

    Examples
    --------
    >>> QNLI = QNLITask()
    >>> QNLI.class_labels
    ['not_entailment', 'entailment']
    >>> type(QNLI.metrics)
    <class 'mxnet.metric.Accuracy'>
    >>> QNLI.dataset_train()[0]
    -etc-
    'train'
    >>> len(QNLI.dataset_train()[1])
    108436
    >>> QNLI.dataset_dev()[0]
    -etc-
    'dev'
    >>> len(QNLI.dataset_dev()[1])
    5732
    >>> QNLI.dataset_test()[0]
    -etc-
    'test'
    >>> len(QNLI.dataset_test()[1])
    5740
    """
    def __init__(self):
        is_pair = True
        class_labels = ['not_entailment', 'entailment']
        metric = Accuracy()
        super(QNLITask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for QNLI.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueQNLI(segment=segment)

class STSBTask(GlueTask):
    """The Sentence Textual Similarity Benchmark task on GlueBenchmark.

    Examples
    --------
    >>> STSB = STSBTask()
    >>> STSB.class_labels
    >>> type(STSB.metrics)
    <class 'mxnet.metric.PearsonCorrelation'>
    >>> STSB.dataset_train()[0]
    -etc-
    'train'
    >>> len(STSB.dataset_train()[1])
    5749
    >>> STSB.dataset_dev()[0]
    -etc-
    'dev'
    >>> len(STSB.dataset_dev()[1])
    1500
    >>> STSB.dataset_test()[0]
    -etc-
    'test'
    >>> len(STSB.dataset_test()[1])
    1379
    """
    def __init__(self):
        is_pair = True
        class_labels = None
        metric = PearsonCorrelation(average='micro')
        super(STSBTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for STSB

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueSTSB(segment=segment)

class CoLATask(GlueTask):
    """The Warstdadt acceptability task on GlueBenchmark.

    Examples
    --------
    >>> CoLA = CoLATask()
    >>> CoLA.class_labels
    ['0', '1']
    >>> type(CoLA.metrics)
    <class 'mxnet.metric.MCC'>
    >>> CoLA.dataset_train()[0]
    -etc-
    'train'
    >>> len(CoLA.dataset_train()[1])
    8551
    >>> CoLA.dataset_dev()[0]
    -etc-
    'dev'
    >>> len(CoLA.dataset_dev()[1])
    1043
    >>> CoLA.dataset_test()[0]
    -etc-
    'test'
    >>> len(CoLA.dataset_test()[1])
    1063
    """
    def __init__(self):
        is_pair = False
        class_labels = ['0', '1']
        metric = MCC(average='micro')
        super(CoLATask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for CoLA

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueCoLA(segment=segment)

class SSTTask(GlueTask):
    """The Stanford Sentiment Treebank task on GlueBenchmark.

    Examples
    --------
    >>> SST = SSTTask()
    >>> SST.class_labels
    ['0', '1']
    >>> type(SST.metrics)
    <class 'mxnet.metric.Accuracy'>
    >>> SST.dataset_train()[0]
    -etc-
    'train'
    >>> len(SST.dataset_train()[1])
    67349
    >>> SST.dataset_dev()[0]
    -etc-
    'dev'
    >>> len(SST.dataset_dev()[1])
    872
    >>> SST.dataset_test()[0]
    -etc-
    'test'
    >>> len(SST.dataset_test()[1])
    1821
    """
    def __init__(self):
        is_pair = False
        class_labels = ['0', '1']
        metric = Accuracy()
        super(SSTTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for SST

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueSST2(segment=segment)

class WNLITask(GlueTask):
    """The Winograd NLI task on GlueBenchmark.

    Examples
    --------
    >>> WNLI = WNLITask()
    >>> WNLI.class_labels
    ['0', '1']
    >>> type(WNLI.metrics)
    <class 'mxnet.metric.Accuracy'>
    >>> WNLI.dataset_train()[0]
    -etc-
    'train'
    >>> len(WNLI.dataset_train()[1])
    635
    >>> WNLI.dataset_dev()[0]
    -etc-
    'dev'
    >>> len(WNLI.dataset_dev()[1])
    71
    >>> WNLI.dataset_test()[0]
    -etc-
    'test'
    >>> len(WNLI.dataset_test()[1])
    146
    """
    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = Accuracy()
        super(WNLITask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for WNLI

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'dev', 'test', 'train'
        """
        return GlueWNLI(segment=segment)

class MNLITask(GlueTask):
    """The Multi-Genre Natural Language Inference task on GlueBenchmark.

    Examples
    --------
    >>> MNLI = MNLITask()
    >>> MNLI.class_labels
    ['neutral', 'entailment', 'contradiction']
    >>> type(MNLI.metrics)
    <class 'mxnet.metric.Accuracy'>
    >>> MNLI.dataset_train()[0]
    -etc-
    'train'
    >>> len(MNLI.dataset_train()[1])
    392702
    >>> MNLI.dataset_dev()[0][0]
    -etc-
    'dev_matched'
    >>> len(MNLI.dataset_dev()[0][1])
    9815
    >>> MNLI.dataset_dev()[1][0]
    'dev_mismatched'
    >>> len(MNLI.dataset_dev()[1][1])
    9832
    >>> MNLI.dataset_test()[0][0]
    -etc-
    'test_matched'
    >>> len(MNLI.dataset_test()[0][1])
    9796
    >>> MNLI.dataset_test()[1][0]
    'test_mismatched'
    >>> len(MNLI.dataset_test()[1][1])
    9847
    """
    def __init__(self):
        is_pair = True
        class_labels = ['neutral', 'entailment', 'contradiction']
        metric = Accuracy()
        super(MNLITask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for MNLI

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'dev_matched', 'dev_mismatched', 'test_matched',
            'test_mismatched', 'train'
        """
        return GlueMNLI(segment=segment)

    def dataset_dev(self):
        """Get the dev segment of the dataset for the task.

        Returns
        -------
        list of TSVDataset : the dataset of the dev segment.
        """
        return [('dev_matched', self.get_dataset(segment='dev_matched')),
                ('dev_mismatched', self.get_dataset(segment='dev_mismatched'))]

    def dataset_test(self):
        """Get the test segment of the dataset for the task.

        Returns
        -------
        list of TSVDataset : the dataset of the test segment.
        """
        return [('test_matched', self.get_dataset(segment='test_matched')),
                ('test_mismatched', self.get_dataset(segment='test_mismatched'))]

class XNLITask(GlueTask):
    """The XNLI task using the dataset released from Baidu

    <https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE>.

    Examples
    --------
    >>> XNLI = XNLITask()
    >>> XNLI.class_labels
    ['neutral', 'entailment', 'contradiction']
    >>> type(XNLI.metrics)
    <class 'mxnet.metric.Accuracy'>
    >>> XNLI.dataset_train()[0]
    'train'
    >>> len(XNLI.dataset_train()[1])
    392702
    >>> XNLI.dataset_dev()[0]
    'dev'
    >>> len(XNLI.dataset_dev()[1])
    2490
    >>> XNLI.dataset_test()[0]
    'test'
    >>> len(XNLI.dataset_test()[1])
    5010
    """
    def __init__(self):
        is_pair = True
        class_labels = ['neutral', 'entailment', 'contradiction']
        metric = Accuracy()
        super(XNLITask, self).__init__(class_labels, metric, is_pair,
                                       label_alias={'contradictory':'contradiction'})

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for XNLI.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'dev', 'test', 'train'
        """
        return BaiduErnieXNLI(segment)

class LCQMCTask(GlueTask):
    """The LCQMC task.

    Note that this dataset is no longer public. You can apply to the dataset owners for LCQMC.
    http://icrc.hitsz.edu.cn/info/1037/1146.htm

    """
    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = Accuracy()
        super(LCQMCTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, file_path, segment='train'):
        # pylint: disable=arguments-differ
        """Get the corresponding dataset for LCQMC.

        Parameters
        ----------
        file_path : str
            Path to the dataset file
        segment : str, default 'train'
            Dataset segments. Options are 'dev', 'test', 'train'
        """
        return BaiduErnieLCQMC(file_path, segment)

class ChnSentiCorpTask(GlueTask):
    """The ChnSentiCorp task using the dataset released from Baidu

    <https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE>.

    Examples
    --------
    >>> ChnSentiCorp = ChnSentiCorpTask()
    >>> ChnSentiCorp.class_labels
    ['0', '1']
    >>> type(ChnSentiCorp.metrics)
    <class 'mxnet.metric.Accuracy'>
    >>> ChnSentiCorp.dataset_train()[0]
    'train'
    >>> len(ChnSentiCorp.dataset_train()[1])
    9600
    >>> ChnSentiCorp.dataset_dev()[0]
    'dev'
    >>> len(ChnSentiCorp.dataset_dev()[1])
    1200
    >>> ChnSentiCorp.dataset_test()[0]
    'test'
    >>> len(ChnSentiCorp.dataset_test()[1])
    1200
    """
    def __init__(self):
        is_pair = False
        class_labels = ['0', '1']
        metric = Accuracy()
        super(ChnSentiCorpTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for ChnSentiCorp.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'dev', 'test', 'train'
        """
        return BaiduErnieChnSentiCorp(segment)

def get_task(task):
    """Returns a pre-defined glue task by name.

    Parameters
    ----------
    task : str
        Options include 'MRPC', 'QNLI', 'RTE', 'STS-B', 'CoLA',
        'MNLI', 'WNLI', 'SST', 'XNLI', 'LCQMC', 'ChnSentiCorp'

    Returns
    -------
    GlueTask
    """
    tasks = {
        'mrpc': MRPCTask(),
        'qqp': QQPTask(),
        'qnli': QNLITask(),
        'rte': RTETask(),
        'sts-b': STSBTask(),
        'cola': CoLATask(),
        'mnli': MNLITask(),
        'wnli': WNLITask(),
        'sst': SSTTask(),
        'xnli': XNLITask(),
        'lcqmc': LCQMCTask(),
        'chnsenticorp': ChnSentiCorpTask()
    }
    if task.lower() not in tasks:
        raise ValueError(
            'Task name %s is not supported. Available options are\n\t%s'%(
                task, '\n\t'.join(sorted(tasks.keys()))))
    return copy(tasks[task.lower()])
