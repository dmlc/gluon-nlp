import argparse
import numpy as np
import time
import random
import os
import mxnet as mx
import logging
from mxnet import gluon
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from mxnet.gluon.data import DataLoader
import gluonnlp.data.batchify as btf
from gluonnlp.data import FixedBucketSampler
from gluonnlp.model.encoder_decoder import get_gnmt_encoder_decoder
from gluonnlp.model.translation import NMTModel
from gluonnlp.data import IWSLT2015
from loss import SoftmaxCEMaskedLoss
from utils import logging_config
import _constants as _C

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description='Neural Machine Translation Example on IWSLT2015.'
                                             'We train the Google NMT model')
parser.add_argument('--num-layers', type=int, default=2, help='number of layers in the encoder'
                                                              ' and decoder')
parser.add_argument('--save-dir', type=str, default='out_dir',
                    help='directory path to save the final model and training log')
parser.add_argument('--num-bi-layers', type=int, default=1,
                    help='number of bidireciontal layers in the encoder')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=1E-3, help='initial learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout applied to layers (0 = no dropout)')
args = parser.parse_args()
print(args)
logging_config(args.save_dir)


def cache_dataset(dataset, prefix):
    """Cache the processed npy dataset  the dataset into a npz

    Parameters
    ----------
    dataset : SimpleDataset
    file_path : str
    """
    if not os.path.exists(_C.CACHE_PATH):
        os.makedirs(_C.CACHE_PATH)
    src_data = np.array([ele[0] for ele in dataset])
    tgt_data = np.array([ele[1] for ele in dataset])
    np.savez(os.path.join(_C.CACHE_PATH, prefix + '.npz'), src_data=src_data, tgt_data=tgt_data)


def load_cached_dataset(prefix):
    cached_file_path = os.path.join(_C.CACHE_PATH, prefix + '.npz')
    if os.path.exists(cached_file_path):
        print('Load cached data from {}'.format(cached_file_path))
        dat = np.load(cached_file_path)
        return ArrayDataset(np.array(dat['src_data']), np.array(dat['tgt_data']))
    else:
        return None


class TrainValDataTransform(object):
    def __init__(self, src_vocab, tgt_vocab):
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab

    def __call__(self, src, tgt):
        src_sentence = self._src_vocab[src.split()]
        tgt_sentence = self._tgt_vocab[tgt.split()]
        src_sentence.append(self._src_vocab[self._src_vocab.eos_token])
        tgt_sentence.insert(0, self._tgt_vocab[self._tgt_vocab.bos_token])
        tgt_sentence.append(self._tgt_vocab[self._tgt_vocab.eos_token])
        src_npy = np.array(src_sentence, dtype=np.int32)
        tgt_npy = np.array(tgt_sentence, dtype=np.int32)
        return src_npy, tgt_npy


def process_dataset(dataset, src_vocab, tgt_vocab):
    start = time.time()
    dataset_processed = dataset.transform(TrainValDataTransform(src_vocab, tgt_vocab), lazy=False)
    end = time.time()
    print('Processing Time spent: {}'.format(end - start))
    return dataset_processed


def load_IWSLT2015(src_lang='en', tgt_lang='vi'):
    common_prefix = 'IWSLT2015_' + src_lang + '_' + tgt_lang
    data_train = IWSLT2015('train', src_lang=src_lang, tgt_lang=tgt_lang)
    data_val = IWSLT2015('val', src_lang=src_lang, tgt_lang=tgt_lang)
    data_test = IWSLT2015('test', src_lang=src_lang, tgt_lang=tgt_lang)
    src_vocab, tgt_vocab = data_train.src_vocab, data_train.tgt_vocab
    data_train_processed = load_cached_dataset(common_prefix + '_train')
    if not data_train_processed:
        data_train_processed = process_dataset(data_train, src_vocab, tgt_vocab)
        cache_dataset(data_train_processed, common_prefix + '_train')
    data_val_processed = load_cached_dataset(common_prefix + '_val')
    if not data_val_processed:
        data_val_processed = process_dataset(data_val, src_vocab, tgt_vocab)
        cache_dataset(data_val_processed, common_prefix + '_val')
    data_val_processed.transform(lambda src, tgt: (src, tgt[:-1], tgt[1:]))
    raw_transform = lambda src, tgt: (src.split(), tgt.split())
    data_val_raw = data_val.transform(raw_transform, lazy=False)
    data_test_raw = data_test.transform(raw_transform, lazy=False)
    return data_train_processed, data_val_processed, data_val_raw, data_test_raw, src_vocab, tgt_vocab


def get_data_lengths(dataset):
    return list(dataset.transform(lambda srg, tgt: (len(srg), len(tgt))))


data_train, data_val, data_val_raw, data_test_raw, src_vocab, tgt_vocab = load_IWSLT2015()
data_train_lengths = get_data_lengths(data_train)
data_val_lengths = get_data_lengths(data_val)
data_train = data_train.transform(lambda src, tgt: (src, tgt[:-1], tgt[1:], len(src), len(tgt) - 1))
data_val = data_val.transform(lambda src, tgt: (src, tgt[:-1], tgt[1:], len(src), len(tgt) - 1))


hidden_size = 128
dropout = 0.2
num_buckets = 5
train_batch_size = 128
test_batch_size = 32
ctx = mx.gpu()

encoder, decoder = get_gnmt_encoder_decoder(hidden_size=hidden_size)
model = NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 embed_size=hidden_size, prefix='gnmt_')
model.initialize(init=mx.init.Xavier(), ctx=ctx)
model.hybridize()

loss_function = SoftmaxCEMaskedLoss()
loss_function.hybridize()

trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr})

batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Pad(), btf.Stack(), btf.Stack())
train_batch_sampler = FixedBucketSampler(lengths=data_train_lengths,
                                         batch_size=train_batch_size,
                                         num_buckets=num_buckets, shuffle=True)
logging.info('Train Batch Sampler:\n{}'.format(train_batch_sampler.stats()))
train_data_loader = DataLoader(data_train,
                               batch_sampler=train_batch_sampler,
                               batchify_fn=batchify_fn,
                               num_workers=4)

val_batch_sampler = FixedBucketSampler(lengths=data_val_lengths,
                                       batch_size=train_batch_size,
                                       num_buckets=num_buckets, shuffle=False)
logging.info('Valid Batch Sampler:\n{}'.format(train_batch_sampler.stats()))
val_data_loader = DataLoader(data_val,
                             batch_sampler=val_batch_sampler,
                             batchify_fn=batchify_fn,
                             num_workers=4)

for batch_id, (src_seq, tgt_seq, gt_seq, src_valid_length, tgt_valid_length)\
        in enumerate(train_data_loader):
    src_seq = mx.nd.array(src_seq, ctx=ctx)
    tgt_seq = mx.nd.array(tgt_seq, ctx=ctx)
    gt_seq = mx.nd.array(gt_seq, ctx=ctx)
    src_valid_length = mx.nd.array(src_valid_length, ctx=ctx)
    tgt_valid_length = mx.nd.array(tgt_valid_length, ctx=ctx)
    with mx.autograd.record():
        out, _ = model(src_seq, tgt_seq, src_valid_length, tgt_valid_length)
        loss = loss_function(out, gt_seq, tgt_valid_length).mean()
        loss.backward()
    grads = [p.grad(ctx) for p in model.collect_params().values()]
    gnorm = gluon.utils.clip_global_norm(grads, args.clip)
    trainer.step(1)
    logging.info('Loss={}, Gradient Norm={}'.format(loss, gnorm))
