"""
Transformer
=================================

This example shows how to implement the Transformer model with Gluon NLP Toolkit.

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones,
          Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6000--6010},
  year={2017}
}
"""

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import time
import random
import os
import zipfile
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.utils import download, check_sha1
import gluonnlp as nlp

from gluonnlp.loss import MaskedSoftmaxCELoss
from gluonnlp.model.translation import NMTModel
from gluonnlp.model.transformer import get_transformer_encoder_decoder
from translation import BeamSearchTranslator
from utils import logging_config
from bleu import _bpe_to_words, compute_bleu
import dataprocessor

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

nlp.utils.check_version('0.7.0')

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We use this script only for transformer inference.')
parser.add_argument('--dataset', type=str, default='WMT2014BPE', help='Dataset to use.')
parser.add_argument('--src_lang', type=str, default='en', help='Source language')
parser.add_argument('--tgt_lang', type=str, default='de', help='Target language')
parser.add_argument('--num_units', type=int, default=512, help='Dimension of the embedding '
                                                               'vectors and states.')
parser.add_argument('--hidden_size', type=int, default=2048,
                    help='Dimension of the hidden state in position-wise feed-forward networks.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--num_layers', type=int, default=6,
                    help='number of layers in the encoder and decoder')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads in multi-head attention')
parser.add_argument('--scaled', action='store_true', help='Turn on to use scale in attention')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size. Number of tokens in a minibatch')
parser.add_argument('--beam_size', type=int, default=4, help='Beam size')
parser.add_argument('--lp_alpha', type=float, default=0.6,
                    help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')
parser.add_argument('--test_batch_size', type=int, default=256, help='Test batch size')
parser.add_argument('--num_buckets', type=int, default=10, help='Bucket number')
parser.add_argument('--bucket_scheme', type=str, default='constant',
                    help='Strategy for generating bucket keys. It supports: '
                         '"constant": all the buckets have the same width; '
                         '"linear": the width of bucket increases linearly; '
                         '"exp": the width of bucket increases exponentially')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the '
                                                                    'throughput of the bucketing')
parser.add_argument('--src_max_len', type=int, default=-1, help='Maximum length of the source '
                                                                'sentence, -1 means no clipping')
parser.add_argument('--tgt_max_len', type=int, default=-1, help='Maximum length of the target '
                                                                'sentence, -1 means no clipping')
parser.add_argument('--full', action='store_true',
                    help='In default, we use the test dataset in'
                         ' http://statmt.org/wmt14/test-filtered.tgz.'
                         ' When the option full is turned on, we use the test dataset in'
                         ' http://statmt.org/wmt14/test-full.tgz')
parser.add_argument('--bleu', type=str, default='tweaked',
                    help='Schemes for computing bleu score. It can be: '
                    '"tweaked": it uses similar steps in get_ende_bleu.sh in tensor2tensor '
                    'repository, where compound words are put in ATAT format; '
                    '"13a": This uses official WMT tokenization and produces the same results'
                    ' as official script (mteval-v13a.pl) used by WMT; '
                    '"intl": This use international tokenization in mteval-v14a.pl')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='transformer_out',
                    help='directory path to save the final model and training log')
parser.add_argument('--gpu', type=int,
                    help='gpu id, e.g. 0 or 1. Unspecified means using cpu.')
parser.add_argument('--model_parameter', type=str, default=' ', required=True,
                    help='model parameter for inference, must be provided.')

args = parser.parse_args()
logging_config(args.save_dir)
logging.info(args)

# data process
data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab \
    = dataprocessor.load_translation_data(dataset=args.dataset, bleu=args.bleu, args=args)

dataprocessor.write_sentences(test_tgt_sentences, os.path.join(args.save_dir, 'test_gt.txt'))

data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                     for i, ele in enumerate(data_val)])
data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                      for i, ele in enumerate(data_test)])

data_train_lengths, data_val_lengths, data_test_lengths = [dataprocessor.get_data_lengths(x)
                                                           for x in
                                                           [data_train, data_val, data_test]]

detokenizer = nlp.data.SacreMosesDetokenizer()

# model prepare
ctx = [mx.cpu()] if args.gpu is None else [mx.gpu(args.gpu)]

if args.src_max_len <= 0 or args.tgt_max_len <= 0:
    max_len = np.max(
        [np.max(data_train_lengths, axis=0), np.max(data_val_lengths, axis=0),
         np.max(data_test_lengths, axis=0)],
        axis=0)

if args.src_max_len > 0:
    src_max_len = args.src_max_len
else:
    src_max_len = max_len[0]
if args.tgt_max_len > 0:
    tgt_max_len = args.tgt_max_len
else:
    tgt_max_len = max_len[1]

encoder, decoder, one_step_ahead_decoder = get_transformer_encoder_decoder(
    units=args.num_units, hidden_size=args.hidden_size, dropout=args.dropout,
    num_layers=args.num_layers, num_heads=args.num_heads, max_src_length=max(src_max_len, 500),
    max_tgt_length=max(tgt_max_len, 500), scaled=args.scaled)
model = NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 one_step_ahead_decoder=one_step_ahead_decoder, share_embed=args.dataset != 'TOY',
                 embed_size=args.num_units, tie_weights=args.dataset != 'TOY',
                 embed_initializer=None, prefix='transformer_')

param_name = args.model_parameter
if (not os.path.exists(param_name)):
    archive_param_url = 'http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/{}'
    archive_file_hash = ('transformer_en_de_512_WMT2014-e25287c5.zip',
                         '5193b469e0e2dfdda3c834f9212420758a0d1d71')
    param_file_hash = ('transformer_en_de_512_WMT2014-e25287c5.params',
                       'e25287c5a924b7025e08d626f02626d5fa3af2d1')
    archive_file, archive_hash = archive_file_hash
    param_file, param_hash = param_file_hash
    logging.warning('The provided param file {} does not exist, start to download it from {}...'
                    .format(param_name, archive_param_url.format(archive_file)))

    root_dir = os.path.dirname(__file__)
    archive_file_path = '{}/{}'.format(root_dir, archive_file)
    param_name = '{}/{}'.format(root_dir, param_file)
    if (not os.path.exists(param_name) or not check_sha1(param_name, param_hash)):
        download(archive_param_url.format(archive_file),
                 path=archive_file_path,
                 sha1_hash=archive_hash)
        with zipfile.ZipFile(archive_file_path) as zf:
            zf.extractall(root_dir)

model.load_parameters(param_name, ctx)

static_alloc = True
model.hybridize(static_alloc=static_alloc)
logging.info(model)

# translator prepare
translator = BeamSearchTranslator(model=model, beam_size=args.beam_size,
                                  scorer=nlp.model.BeamSearchScorer(alpha=args.lp_alpha,
                                                                    K=args.lp_k),
                                  max_length=200)
logging.info('Use beam_size={}, alpha={}, K={}'.format(args.beam_size, args.lp_alpha, args.lp_k))

test_loss_function = MaskedSoftmaxCELoss()
test_loss_function.hybridize(static_alloc=static_alloc)

def inference():
    """inference function."""
    logging.info('Inference on test_dataset!')

    # data prepare
    test_data_loader = dataprocessor.get_dataloader(data_test, args,
                                                    dataset_type='test',
                                                    use_average_length=True)

    if args.bleu == 'tweaked':
        bpe = bool(args.dataset != 'IWSLT2015' and args.dataset != 'TOY')
        split_compound_word = bpe
        tokenized = True
    elif args.bleu == '13a' or args.bleu == 'intl':
        bpe = False
        split_compound_word = False
        tokenized = False
    else:
        raise NotImplementedError

    translation_out = []
    all_inst_ids = []
    total_wc = 0
    total_time = 0
    batch_total_blue = 0

    for batch_id, (src_seq, tgt_seq, src_test_length, tgt_test_length, inst_ids) \
            in enumerate(test_data_loader):

        total_wc += src_test_length.sum().asscalar() + tgt_test_length.sum().asscalar()

        src_seq = src_seq.as_in_context(ctx[0])
        tgt_seq = tgt_seq.as_in_context(ctx[0])
        src_test_length = src_test_length.as_in_context(ctx[0])
        tgt_test_length = tgt_test_length.as_in_context(ctx[0])
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())

        start = time.time()
        # Translate to get a bleu score
        samples, _, sample_test_length = \
            translator.translate(src_seq=src_seq, src_valid_length=src_test_length)
        total_time += (time.time() - start)

        # generator the translator result for each batch
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_test_length = sample_test_length[:, 0].asnumpy()
        translation_tmp = []
        translation_tmp_sentences = []
        for i in range(max_score_sample.shape[0]):
            translation_tmp.append([tgt_vocab.idx_to_token[ele] for ele in \
                                    max_score_sample[i][1:(sample_test_length[i] - 1)]])

        # detokenizer each translator result
        for _, sentence in enumerate(translation_tmp):
            if args.bleu == 'tweaked':
                translation_tmp_sentences.append(sentence)
                translation_out.append(sentence)
            elif args.bleu == '13a' or args.bleu == 'intl':
                translation_tmp_sentences.append(detokenizer(_bpe_to_words(sentence)))
                translation_out.append(detokenizer(_bpe_to_words(sentence)))
            else:
                raise NotImplementedError

        # generate tgt_sentence for bleu calculation of each batch
        tgt_sen_tmp = [test_tgt_sentences[index] for \
                         _, index in enumerate(inst_ids.asnumpy().astype(np.int32).tolist())]
        batch_test_bleu_score, _, _, _, _ = compute_bleu([tgt_sen_tmp], translation_tmp_sentences,
                                                         tokenized=tokenized, tokenizer=args.bleu,
                                                         split_compound_word=split_compound_word,
                                                         bpe=bpe)
        batch_total_blue += batch_test_bleu_score

        # log for every ten batchs
        if batch_id % 10 == 0 and batch_id != 0:
            batch_ave_bleu = batch_total_blue / 10
            batch_total_blue = 0
            logging.info('batch id={:d}, batch_bleu={:.4f}'
                         .format(batch_id, batch_ave_bleu * 100))

    # reorg translation sentences by inst_ids
    real_translation_out = [None for _ in range(len(all_inst_ids))]
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = sentence

    # get bleu score, n-gram precisions, brevity penalty,  reference length, and translation length
    test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], real_translation_out,
                                               tokenized=tokenized, tokenizer=args.bleu,
                                               split_compound_word=split_compound_word,
                                               bpe=bpe)

    logging.info('Inference at test dataset. \
                 inference bleu={:.4f}, throughput={:.4f}K wps'
                 .format(test_bleu_score * 100, total_wc / total_time / 1000))


if __name__ == '__main__':
    inference()
