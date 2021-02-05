import gluonnlp
import numpy as np
import mxnet as mx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from mxnet.gluon import nn
from mxnet.gluon.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric
from gluonnlp.models import get_backbone
from gluonnlp.utils.parameter import clip_grad_global_norm
from gluonnlp.utils.preprocessing import get_trimmed_lengths
from gluonnlp.utils.misc import get_mxnet_visible_ctx, grouper, repeat
from mxnet.gluon.data import batchify as bf
from mxnet.gluon.data import DataLoader
from mxnet.lr_scheduler import PolyScheduler
from gluonnlp.utils import set_seed

__all__ = ['get_task']


_FEATURE_COLUMN = {'sst':'sentence',
                   'sts':['sentence1', 'sentence2'],
                   'cola':'sentence',
                   'mrpc':['sentence1', 'sentence2'],
                   'wnli':['sentence1', 'sentence2'],
                   'qnli':['sentence1', 'sentence2'],
                   'qqp':['sentence1', 'sentence2'],
                   'rte':['sentence1', 'sentence2'],
                   'mnli':['sentence1', 'sentence2']
                   }
_LABEL_COLUMN = { 'sst':'label',
                  'sts':'score',
                  'cola':'label',
                  'mrpc':'label',
                  'qnli':'label',
                  'wnli':'label',
                  'qqp':'label',
                  'rte':'label',
                  'mnli':'label',
                  }
_METRIC = {'mrpc':['Accuracy', 'f1'],
           'qqp':['Accuracy', 'f1'],
           'rte':['Accuracy'],
           'qnli':['Accuracy'],
           'sts':['PearsonCorrelation'],
           'cola':['mcc'],
           'sst':['Accuracy'],
           'mnli':['Accuracy'],
           }

class GlueTask:
    def __init__(self, task_name, class_num, feature_column, label_column, metric):
        self.task_name = task_name
        self.class_num = class_num
        self.feature_column = feature_column
        self.label_column = label_column
        self.metric = metric


class SST(GlueTask):
    def __init__(self):
        task_name = 'sst'
        class_num = 2
        feature_column = 'sentence'
        label_column = 'label'
        metric = [Accuracy()]
        super(SST, self).__init__(task_name, class_num, feature_column, label_column, metric)

class STS(GlueTask):
    def __init__(self):
        task_name = 'sts'
        class_num = 1
        feature_column = ['sentence1', 'sentence2']
        label_column = 'score'
        metric = [PearsonCorrelation()]
        super(STS, self).__init__(task_name, class_num, feature_column, label_column, metric)

class COLA(GlueTask):
    def __init__(self):
        task_name = 'cola'
        class_num = 2
        feature_column = 'sentence'
        label_column = 'label'
        metric = [MCC()]
        super(COLA, self).__init__(task_name, class_num, feature_column, label_column, metric)

class MRPC(GlueTask):
    def __init__(self):
        task_name = 'mrpc'
        class_num = 2
        feature_column = ['sentence1', 'sentence2']
        label_column = 'label'
        metric = [Accuracy(), F1()]
        super(MRPC, self).__init__(task_name, class_num, feature_column, label_column, metric)

class QQP(GlueTask):
    def __init__(self):
        task_name = 'qqp'
        class_num = 2
        feature_column = ['sentence1', 'sentence2']
        label_column = 'label'
        metric = [Accuracy()]
        super(QQP, self).__init__(task_name, class_num, feature_column, label_column, metric)

class RTE(GlueTask):
    def __init__(self):
        task_name = 'rte'
        class_num = 2
        feature_column = ['sentence1', 'sentence2']
        label_column = 'label'
        metric = [Accuracy()]
        super(RTE, self).__init__(task_name, class_num, feature_column, label_column, metric)

class MNLI(GlueTask):
    def __init__(self):
        task_name = 'mnli'
        class_num = 3
        feature_column = ['sentence1', 'sentence2']
        label_column = 'label'
        metric = [Accuracy()]
        super(MNLI, self).__init__(task_name, class_num, feature_column, label_column, metric)


def get_task(task_name):
    tasks = { 'mnli':MNLI(),
              'sst':SST(),
              'sts':STS(),
              'qqp':QQP(),
              'rte':RTE(),
              'mrpc':MRPC(),
              'cola':COLA(),
              }

    if task_name.lower() not in tasks:
        raise ValueError(
            'Task name %s is not supported. Available options are\n\t%s' % (
                task_name, '\n\t'.join(sorted(tasks.keys()))))

    return copy.copy(tasks[task_name.lower()])