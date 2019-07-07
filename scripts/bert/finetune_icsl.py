"""
Intent Classification and Slot Labelling with BERT

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
joint intent classification and slot labelling, with Gluon NLP Toolkit.

"""

# coding: utf-8

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
# pylint:disable=redefined-outer-name,logging-format-interpolation,arguments-differ,unused-variable,missing-docstring,wrong-import-order
import os
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, Block
from seqeval.metrics import f1_score as ner_f1_score
import gluonnlp as nlp
from gluonnlp.data import BERTTokenizer, ATISDataset, SNIPSDataset



class BERTForICSL(Block):
    """Model

    """
    def __init__(self, bert, num_intent_classes, num_slot_classes, dropout_prob,
                 prefix=None, params=None):
        """

        Parameters
        ----------
        bert : Block
        num_intent_classes : int
        num_slot_classes : int
        dropout_prob : float
        prefix : None or str
        params : None or ParamDict
        """
        super(BERTForICSL, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.intent_classifier = nn.HybridSequential()
            with self.intent_classifier.name_scope():
                self.intent_classifier.add(nn.Dropout(rate=dropout_prob))
                self.intent_classifier.add(nn.Dense(units=num_intent_classes, flatten=False))
            self.slot_tagger = nn.HybridSequential()
            with self.slot_tagger.name_scope():
                self.slot_tagger.add(nn.Dropout(rate=dropout_prob))
                self.slot_tagger.add(nn.Dense(units=num_slot_classes, flatten=False))

    def forward(self, inputs, valid_length):
        """

        Parameters
        ----------
        inputs : NDArray
            The input sentences, has shape (batch_size, seq_length)
        valid_length : NDArray
            The valid length of the sentences

        Returns
        -------
        intent_scores : NDArray
            Shape (batch_size, num_classes)
        slot_scores : NDArray
            Shape (batch_size, seq_length, num_tag_types)
        """
        token_types = mx.nd.zeros_like(inputs)
        encoded_states, pooler_out = self.bert(inputs, token_types, valid_length)
        intent_scores = self.intent_classifier(pooler_out)
        slot_scores = self.slot_tagger(encoded_states)
        return intent_scores, slot_scores


class IDSLSubwordTransform():
    """Transform the word_tokens/tags by the subword tokenizer

    """
    def __init__(self, subword_vocab, subword_tokenizer, slot_vocab, cased=False):
        """

        Parameters
        ----------
        subword_vocab : Vocab
        subword_tokenizer : Tokenizer
        cased : bool
            Whether to convert all characters to lower
        """
        self._subword_vocab = subword_vocab
        self._subword_tokenizer = subword_tokenizer
        self._slot_vocab = slot_vocab
        self._cased = cased
        self._slot_pad_id = self._slot_vocab['O']


    def __call__(self, word_tokens, tags, intent_ids):
        """ Transform the word_tokens/tags by the subword tokenizer

        Parameters
        ----------
        word_tokens : List[str]
        tags : List[str]
        intent_ids : np.ndarray

        Returns
        -------
        subword_ids : np.ndarray
        subword_mask : np.ndarray
        selected : np.ndarray
        padded_tag_ids : np.ndarray
        intent_label : int
        length : int
        """
        subword_ids = []
        subword_mask = []
        selected = []
        padded_tag_ids = []
        intent_label = intent_ids[0]
        ptr = 0
        for token, tag in zip(word_tokens, tags):
            if not self._cased:
                token = token.lower()
            token_sw_ids = self._subword_vocab[self._subword_tokenizer(token)]
            subword_ids.extend(token_sw_ids)
            subword_mask.extend([1] + [0] * (len(token_sw_ids) - 1))
            selected.append(ptr)
            padded_tag_ids.extend([self._slot_vocab[tag]] +
                                  [self._slot_pad_id] * (len(token_sw_ids) - 1))
            ptr += len(token_sw_ids)
        length = len(subword_ids)
        if len(subword_ids) != len(padded_tag_ids):
            print(word_tokens)
            print(tags)
            print(subword_ids)
            print(padded_tag_ids)
        return np.array(subword_ids, dtype=np.int32),\
               np.array(subword_mask, dtype=np.int32),\
               np.array(selected, dtype=np.int32),\
               np.array(padded_tag_ids, dtype=np.int32),\
               intent_label,\
               length


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='Train a BERT-based model for joint intent detection and slot filling on '
                    'ATIS/SNIPS dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('--seed', type=int, default=123)
    arg_parser.add_argument('--dataset', choices=['atis', 'snips'], default='atis')
    arg_parser.add_argument('--bert-model', type=str, default='bert_12_768_12',
                            help='Name of the BERT model')
    arg_parser.add_argument('--cased', action='store_true',
                            help='Whether to use the cased model trained on book_corpus_wiki_en.'
                                 'Otherwise, use the uncased model.')
    arg_parser.add_argument('--dropout-prob', type=float, default=0.1,
                            help='Dropout probability for the last layer')
    arg_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    arg_parser.add_argument('--epochs', type=int, default=40, help='Batch size for training')
    arg_parser.add_argument('--optimizer', type=str, default='bertadam',
                            help='Optimization algorithm to use')
    arg_parser.add_argument('--learning-rate', type=float, default=5e-5,
                            help='Learning rate for optimization')
    arg_parser.add_argument('--wd', type=float, default=0.0,
                            help='Weight decay')
    arg_parser.add_argument('--warmup-ratio', type=float, default=0.1,
                            help='Warmup ratio for learning rate scheduling')
    arg_parser.add_argument('--slot-loss-mult', type=float, default=1.0,
                            help='Multiplier for the slot loss.')
    arg_parser.add_argument('--save-dir', type=str, default='saved_model')
    arg_parser.add_argument('--gpu', type=int, default=None,
                            help='Number (index) of GPU to run on, e.g. 0.'
                                 ' If not specified, uses CPU.')
    args = arg_parser.parse_args()
    return args



def print_sample(dataset, sample_id):
    """ Print sample in the dataset

    Parameters
    ----------
    dataset : SimpleDataset
    sample_id: int

    Returns
    -------
    """
    word_tokens, tags, intent_ids = dataset[sample_id]
    print('Sample #ID: {} Intent: {}'.format(sample_id,
                                             [dataset.intent_vocab.idx_to_token[ele]
                                              for ele in intent_ids]))
    df = pd.DataFrame(list(zip(word_tokens, tags)))
    df.index.name = None
    print('Sequence:')
    print(df.to_string(header=False))


def evaluation(ctx, data_loader, net, intent_pred_loss, slot_pred_loss, slot_vocab):
    """ Evaluate the trained model

    Parameters
    ----------
    ctx : Context
    data_loader : DataLoader
    net : Block
    intent_pred_loss : Loss
    slot_pred_loss : Loss
    slot_vocab : Vocab

    Returns
    -------
    avg_intent_loss : float
    avg_slot_loss : float
    intent_acc : float
    slot_f1 : float
    pred_slots : list
    gt_slots : list
    """
    nsample = 0
    nslot = 0
    avg_intent_loss = 0
    avg_slot_loss = 0
    correct_intent = 0
    pred_slots = []
    gt_slots = []
    for token_ids, mask, selected, slot_ids, intent_label, valid_length in data_loader:
        token_ids = mx.nd.array(token_ids, ctx=ctx).astype(np.int32)
        mask = mx.nd.array(mask, ctx=ctx).astype(np.float32)
        slot_ids = mx.nd.array(slot_ids, ctx=ctx).astype(np.int32)
        intent_label = mx.nd.array(intent_label, ctx=ctx).astype(np.int32)
        valid_length = mx.nd.array(valid_length, ctx=ctx).astype(np.float32)
        batch_nslot = mask.sum().asscalar()
        batch_nsample = token_ids.shape[0]
        # Forward network
        intent_scores, slot_scores = net(token_ids, valid_length)
        intent_loss = intent_pred_loss(intent_scores, intent_label)
        slot_loss = slot_pred_loss(slot_scores, slot_ids, mask.expand_dims(axis=-1))
        avg_intent_loss += intent_loss.sum().asscalar()
        avg_slot_loss += slot_loss.sum().asscalar()
        pred_slot_ids = mx.nd.argmax(slot_scores, axis=-1).astype(np.int32)
        correct_intent += (mx.nd.argmax(intent_scores, axis=-1).astype(np.int32)
                           == intent_label).sum().asscalar()
        for i in range(batch_nsample):
            ele_valid_length = int(valid_length[i].asscalar())
            ele_sel = selected[i].asnumpy()[:ele_valid_length]
            ele_gt_slot_ids = slot_ids[i].asnumpy()[ele_sel]
            ele_pred_slot_ids = pred_slot_ids[i].asnumpy()[ele_sel]
            ele_gt_slot_tokens = [slot_vocab.idx_to_token[v] for v in ele_gt_slot_ids]
            ele_pred_slot_tokens = [slot_vocab.idx_to_token[v] for v in ele_pred_slot_ids]
            gt_slots.append(ele_gt_slot_tokens)
            pred_slots.append(ele_pred_slot_tokens)
        nsample += batch_nsample
        nslot += batch_nslot
    avg_intent_loss /= nsample
    avg_slot_loss /= nslot
    intent_acc = correct_intent / float(nsample)
    slot_f1 = ner_f1_score(pred_slots, gt_slots)
    return avg_intent_loss, avg_slot_loss, intent_acc, slot_f1, pred_slots, gt_slots



def train(args):
    ctx = mx.cpu() if args.gpu is None else mx.gpu(args.gpu)
    dataset_name = 'book_corpus_wiki_en_cased' if args.cased else 'book_corpus_wiki_en_uncased'
    bert_model, bert_vocab = nlp.model.get_model(name=args.bert_model,
                                                 dataset_name=dataset_name,
                                                 pretrained=True,
                                                 ctx=ctx,
                                                 use_pooler=True,
                                                 use_decoder=False,
                                                 use_classifier=False,
                                                 dropout=args.dropout_prob,
                                                 embed_dropout=args.dropout_prob)
    tokenizer = BERTTokenizer(bert_vocab, lower=not args.cased)
    if args.dataset == 'atis':
        train_data = ATISDataset('train')
        dev_data = ATISDataset('dev')
        test_data = ATISDataset('test')
        intent_vocab = train_data.intent_vocab
        slot_vocab = train_data.slot_vocab
    elif args.dataset == 'snips':
        train_data = SNIPSDataset('train')
        dev_data = SNIPSDataset('dev')
        test_data = SNIPSDataset('test')
        intent_vocab = train_data.intent_vocab
        slot_vocab = train_data.slot_vocab
    else:
        raise NotImplementedError
    print('Dataset {}'.format(args.dataset))
    print('   #Train/Dev/Test = {}/{}/{}'.format(len(train_data), len(dev_data), len(test_data)))
    print('   #Intent         = {}'.format(len(intent_vocab)))
    print('   #Slot           = {}'.format(len(slot_vocab)))
    # Display An Example
    print('Display A Samples')
    print_sample(test_data, 1)
    print('-' * 80)

    idsl_transform = IDSLSubwordTransform(subword_vocab=bert_vocab,
                                          subword_tokenizer=tokenizer,
                                          slot_vocab=slot_vocab,
                                          cased=args.cased)
    train_data_bert = train_data.transform(idsl_transform, lazy=False)
    dev_data_bert = dev_data.transform(idsl_transform, lazy=False)
    test_data_bert = test_data.transform(idsl_transform, lazy=False)
    # Construct the DataLoader
    batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(),    # Subword ID
                                          nlp.data.batchify.Pad(),    # Subword Mask
                                          nlp.data.batchify.Pad(),    # Beginning of subword
                                          nlp.data.batchify.Pad(),    # Tag IDs
                                          nlp.data.batchify.Stack(),  # Intent Label
                                          nlp.data.batchify.Stack())  # Valid Length
    train_batch_sampler = nlp.data.sampler.SortedBucketSampler(
        [len(ele) for ele in train_data_bert],
        batch_size=args.batch_size,
        mult=20,
        shuffle=True)
    train_loader = gluon.data.DataLoader(dataset=train_data_bert,
                                         num_workers=4,
                                         batch_sampler=train_batch_sampler,
                                         batchify_fn=batchify_fn)
    dev_loader = gluon.data.DataLoader(dataset=dev_data_bert,
                                       num_workers=4,
                                       batch_size=args.batch_size,
                                       batchify_fn=batchify_fn,
                                       shuffle=False)
    test_loader = gluon.data.DataLoader(dataset=test_data_bert,
                                        num_workers=4,
                                        batch_size=args.batch_size,
                                        batchify_fn=batchify_fn,
                                        shuffle=False)

    # Build the network and loss functions
    intent_pred_loss = gluon.loss.SoftmaxCELoss()
    slot_pred_loss = gluon.loss.SoftmaxCELoss(batch_axis=[0, 1])

    net = BERTForICSL(bert_model, num_intent_classes=len(intent_vocab),
                      num_slot_classes=len(slot_vocab), dropout_prob=args.dropout_prob)
    net.slot_tagger.initialize(ctx=ctx, init=mx.init.Normal(0.02))
    net.intent_classifier.initialize(ctx=ctx, init=mx.init.Normal(0.02))
    net.hybridize()
    intent_pred_loss.hybridize()
    slot_pred_loss.hybridize()

    # Build the trainer
    trainer = gluon.Trainer(net.collect_params(), args.optimizer,
                            {'learning_rate': args.learning_rate, 'wd': args.wd},
                            update_on_kvstore=False)

    step_num = 0
    num_train_steps = int(len(train_batch_sampler) * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_ratio)
    best_dev_sf1 = -1
    for epoch_id in range(args.epochs):
        avg_train_intent_loss = 0.0
        avg_train_slot_loss = 0.0
        nsample = 0
        nslot = 0
        ntoken = 0
        train_epoch_start = time.time()
        for token_ids, mask, _, slot_ids, intent_label, valid_length\
                in tqdm(train_loader, file=sys.stdout):
            ntoken += valid_length.sum().asscalar()
            token_ids = mx.nd.array(token_ids, ctx=ctx).astype(np.int32)
            mask = mx.nd.array(mask, ctx=ctx).astype(np.float32)
            slot_ids = mx.nd.array(slot_ids, ctx=ctx).astype(np.int32)
            intent_label = mx.nd.array(intent_label, ctx=ctx).astype(np.int32)
            valid_length = mx.nd.array(valid_length, ctx=ctx).astype(np.float32)
            batch_nslots = mask.sum().asscalar()
            batch_nsample = token_ids.shape[0]

            # Set learning rate warm-up
            step_num += 1
            if step_num < num_warmup_steps:
                new_lr = args.learning_rate * step_num / num_warmup_steps
            else:
                offset = ((step_num - num_warmup_steps) * args.learning_rate /
                          (num_train_steps - num_warmup_steps))
                new_lr = args.learning_rate - offset
            trainer.set_learning_rate(new_lr)

            with mx.autograd.record():
                intent_scores, slot_scores = net(token_ids, valid_length)
                intent_loss = intent_pred_loss(intent_scores, intent_label)
                slot_loss = slot_pred_loss(slot_scores, slot_ids, mask.expand_dims(axis=-1))
                intent_loss = intent_loss.mean()
                slot_loss = slot_loss.sum() / batch_nslots
                loss = intent_loss + args.slot_loss_mult * slot_loss
                loss.backward()
            trainer.update(1.0)
            avg_train_intent_loss += intent_loss.asscalar() * batch_nsample
            avg_train_slot_loss += slot_loss.asscalar() * batch_nslots
            nsample += batch_nsample
            nslot += batch_nslots
        train_epoch_end = time.time()
        avg_train_intent_loss /= nsample
        avg_train_slot_loss /= nslot
        print('[Epoch {}] train intent/slot = {:.3f}/{:.3f}, #token per second={:.0f}'.format(
            epoch_id, avg_train_intent_loss, avg_train_slot_loss,
            ntoken / (train_epoch_end - train_epoch_start)))
        avg_dev_intent_loss, avg_dev_slot_loss, dev_intent_acc,\
        dev_slot_f1, dev_pred_slots, dev_gt_slots\
            = evaluation(ctx, dev_loader, net, intent_pred_loss, slot_pred_loss, slot_vocab)
        print('[Epoch {}]    dev intent/slot = {:.3f}/{:.3f},'
              ' slot f1 = {:.2f}, intent acc = {:.2f}'.format(epoch_id, avg_dev_intent_loss,
                                                              avg_dev_slot_loss,
                                                              dev_slot_f1 * 100,
                                                              dev_intent_acc * 100))
        if dev_slot_f1 > best_dev_sf1:
            best_dev_sf1 = dev_slot_f1
            avg_test_intent_loss, avg_test_slot_loss, test_intent_acc, \
            test_slot_f1, test_pred_slots, test_gt_slots \
                = evaluation(ctx, test_loader, net, intent_pred_loss, slot_pred_loss, slot_vocab)
            print('[Epoch {}]    test intent/slot = {:.3f}/{:.3f},'
                  ' slot f1 = {:.2f}, intent acc = {:.2f}'.format(epoch_id, avg_test_intent_loss,
                                                                  avg_test_slot_loss,
                                                                  test_slot_f1 * 100,
                                                                  test_intent_acc * 100))
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            net.save_parameters(os.path.join(args.save_dir, 'best_valid.params'))
    print('Evaluate the best model:')
    net.load_parameters(os.path.join(args.save_dir, 'best_valid.params'))
    avg_test_intent_loss, avg_test_slot_loss, test_intent_acc, \
    test_slot_f1, test_pred_slots, test_gt_slots \
        = evaluation(ctx, test_loader, net, intent_pred_loss, slot_pred_loss, slot_vocab)
    print('Best validation model --> Slot F1={:.2f}, Intent acc={:.2f}'
          .format(test_slot_f1 * 100, test_intent_acc * 100))
    with open(os.path.join(args.save_dir, 'test_error.txt'), 'w') as of:
        of.write('{} {}\n'.format(test_slot_f1, test_intent_acc))

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    random.seed(args.seed)
    train(args)
