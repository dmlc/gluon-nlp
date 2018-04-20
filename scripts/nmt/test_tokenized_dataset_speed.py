from gluonnlp.model.encoder_decoder import get_gnmt_encoder_decoder
from gluonnlp.model.translation import NMTModel
import numpy as np
import time
import multiprocessing as mp
from gluonnlp import Vocab
from gluonnlp.data.utils import Counter
from gluonnlp.data import WMT2016BPE
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from dataset import get_tokenized_dataset, TokenizedDataset

data_train_text_line = WMT2016BPE('train')
#
# tokenized_dat_src = get_tokenized_dataset([ele[0] for ele in data_train])
# tokenized_dat_tgt = get_tokenized_dataset([ele[1] for ele in data_train])
#
# tokenized_dat_src.save('wmt2016_train_tokenized_src.npz')
# tokenized_dat_tgt.save('wmt2016_train_tokenized_tgt.npz')
#


def load_dataset():
    tokenized_dat_src = TokenizedDataset.load('wmt2016_train_tokenized_src.npz')
    tokenized_dat_tgt = TokenizedDataset.load('wmt2016_train_tokenized_tgt.npz')
    return ArrayDataset(tokenized_dat_src, tokenized_dat_tgt)


def process_data(ele):
    src_sentence = src_vocab[ele[0]]
    tgt_sentence = tgt_vocab[ele[1]]
    src_sentence.append(src_vocab[src_vocab.eos_token])
    tgt_sentence.insert(0, tgt_vocab[tgt_vocab.bos_token])
    tgt_sentence.append(tgt_vocab[tgt_vocab.eos_token])
    src_npy = np.array(src_sentence, dtype=np.int32)
    tgt_npy = np.array(tgt_sentence, dtype=np.int32)
    return src_npy, tgt_npy


data_train = load_dataset()
src_vocab, tgt_vocab = data_train_text_line.src_vocab, data_train_text_line.tgt_vocab

start = time.time()
with mp.Pool(4) as pool:
    data_train = SimpleDataset(pool.map(process_data, data_train))
end = time.time()
