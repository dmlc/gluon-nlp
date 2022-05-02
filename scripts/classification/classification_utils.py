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
from gluonnlp.utils.misc import get_mxnet_visible_device, grouper, repeat
from mxnet.gluon.data import batchify as bf
from mxnet.gluon.data import DataLoader
from mxnet.lr_scheduler import PolyScheduler
from gluonnlp.utils import set_seed

__all__ = ['get_task']



def get_proj(label):
    label_proj = {}
    label_names = []
    for i in range(len(label)):
        if label[i] not in label_names:
            label_names.append(label[i])
    label_names.sort()
    for i in range(len(label_names)):
        label_proj[label_names[i]] = i
    return label_proj

class GlueTask:
    def __init__(self, task_name, class_num, feature_column, label_column, metric, train_dir=None, eval_dir=None):
        self.task_name = task_name
        self.class_num = class_num
        self.feature_column = feature_column
        self.label_column = label_column
        self.metric = metric
        self.raw_eval_data = None
        self.raw_train_data = None
        self.proj_label = {}

    def set_dataset(self, train_dir, eval_dir):
        if train_dir:
            self.raw_train_data = pd.read_parquet(train_dir)
        if eval_dir:
            self.raw_eval_data = pd.read_parquet(eval_dir)
        if self.task_name == 'sts':
            pass
        else:
            if train_dir:
                self.proj_label = get_proj(self.raw_train_data[self.label_column])
            elif eval_dir:
                self.proj_label = get_proj(self.raw_eval_data[self.label_column])
            else:
                raise ValueError('Need at least one input' )



class SST(GlueTask):
    def __init__(self, train_dir=None, eval_dir=None):
        task_name = 'sst'
        class_num = 2
        feature_column = 'sentence'
        label_column = 'label'
        metric = [Accuracy()]
        super(SST, self).__init__(task_name, class_num, feature_column, label_column, metric, train_dir, eval_dir)

class STS(GlueTask):
    def __init__(self, train_dir=None, eval_dir=None):
        task_name = 'sts'
        class_num = 1
        feature_column = ['sentence1', 'sentence2']
        label_column = 'score'
        metric = [PearsonCorrelation()]
        super(STS, self).__init__(task_name, class_num, feature_column, label_column, metric, train_dir, eval_dir)

class COLA(GlueTask):
    def __init__(self, train_dir=None, eval_dir=None):
        task_name = 'cola'
        class_num = 2
        feature_column = 'sentence'
        label_column = 'label'
        metric = [MCC()]
        super(COLA, self).__init__(task_name, class_num, feature_column, label_column, metric, train_dir, eval_dir)

class MRPC(GlueTask):
    def __init__(self, train_dir=None, eval_dir=None):
        task_name = 'mrpc'
        class_num = 2
        feature_column = ['sentence1', 'sentence2']
        label_column = 'label'
        metric = [Accuracy(), F1()]
        super(MRPC, self).__init__(task_name, class_num, feature_column, label_column, metric, train_dir, eval_dir)

class QQP(GlueTask):
    def __init__(self, train_dir=None, eval_dir=None):
        task_name = 'qqp'
        class_num = 2
        feature_column = ['sentence1', 'sentence2']
        label_column = 'label'
        metric = [Accuracy()]
        super(QQP, self).__init__(task_name, class_num, feature_column, label_column, metric, train_dir, eval_dir)

class RTE(GlueTask):
    def __init__(self, train_dir=None, eval_dir=None):
        task_name = 'rte'
        class_num = 2
        feature_column = ['sentence1', 'sentence2']
        label_column = 'label'
        metric = [Accuracy()]
        super(RTE, self).__init__(task_name, class_num, feature_column, label_column, metric, train_dir, eval_dir)

class MNLI(GlueTask):
    def __init__(self, train_dir=None, eval_dir=None):
        task_name = 'mnli'
        class_num = 3
        feature_column = ['sentence1', 'sentence2']
        label_column = 'label'
        metric = [Accuracy()]
        super(MNLI, self).__init__(task_name, class_num, feature_column, label_column, metric, train_dir, eval_dir)


def get_task(task_name, train_dir=None, eval_dir=None):
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
    task = tasks[task_name.lower()]
    task.set_dataset(train_dir, eval_dir)
    return copy.copy(task)

