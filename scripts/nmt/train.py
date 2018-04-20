from gluonnlp.model.encoder_decoder import get_gnmt_encoder_decoder
from gluonnlp.model.translation import NMTModel
import numpy as np
import time
import os
import multiprocessing as mp
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from gluonnlp import Vocab
from gluonnlp.data.utils import Counter
from gluonnlp.data import IWSLT2015
import _constants as _C


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
        dat = np.load(cached_file_path)
        return ArrayDataset(np.array(dat['src_data']), np.array(dat['tgt_data']))
    else:
        return None

class TrainValDataTransform(object):
    def __init__(self, src_vocab, tgt_vocab):
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab

    def __call__(self, ele):
        src_sentence = self._src_vocab[ele[0].split()]
        tgt_sentence = self._tgt_vocab[ele[1].split()]
        src_sentence.append(self._src_vocab[self._src_vocab.eos_token])
        tgt_sentence.insert(0, self._tgt_vocab[self._tgt_vocab.bos_token])
        tgt_sentence.append(self._tgt_vocab[self._tgt_vocab.eos_token])
        src_npy = np.array(src_sentence, dtype=np.int32)
        tgt_npy = np.array(tgt_sentence, dtype=np.int32)
        return src_npy, tgt_npy


def process_dataset(dataset, src_vocab, tgt_vocab):
    start = time.time()
    with mp.Pool(4) as pool:
        dataset_processed = \
            SimpleDataset(pool.map(TrainValDataTransform(src_vocab, tgt_vocab),
                                   dataset))
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
    if not data_train_processed:
        data_val_processed = process_dataset(data_val, src_vocab, tgt_vocab)
        cache_dataset(data_val_processed, common_prefix + '_val')
    data_test_processed = data_test.transform(lambda ele: ele.split, lazy=False)
    return data_train_processed, data_val_processed, data_test_processed

data_train, data_val, data_test = load_IWSLT2015()
