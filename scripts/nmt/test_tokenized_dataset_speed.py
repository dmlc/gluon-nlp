from gluonnlp.model.encoder_decoder import get_gnmt_encoder_decoder
from gluonnlp.model.translation import NMTModel
import numpy as np
import time
from gluonnlp import Vocab
from gluonnlp.data.utils import Counter
from gluonnlp.data import WMT2016BPE
from dataset import get_tokenized_dataset

data_train = WMT2016BPE('train')

tokenized_dat_src = get_tokenized_dataset([ele[0] for ele in data_train])
tokenized_dat_tgt = get_tokenized_dataset([ele[1] for ele in data_train])

tokenized_dat_src.save('wmt2016_train_tokenized_src.npz')
tokenized_dat_tgt.save('wmt2016_train_tokenized_tgt.npz')

