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

# Some functions are used to aid the training model
# @author：kenjewu
# @date：2018/12/12


import os
import logging
import numpy as np

from mxboard import SummaryWriter
from mxnet import autograd, gluon, nd


def conll_evaluate(output_file, score_file):
    '''
    use conll2003evaluate shell to evaluate the perfomance of algorithm

    Args:
        output_file (str): path of model predict result
        score_file (str): path of socre result

    Returns:
        acc (float): accuracy
        precision (float): precision
        recall (float): recall
        f1 (float): F1 score
    '''

    os.system("./conll2003eval2.sh < %s > %s" % (output_file, score_file))
    with open(score_file, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1


def clip_gradients(model, clip, ctx):
    '''clip the gradient of params

    Args:
        model (Block): model
        clip (float): clip value
        ctx (ctx): context
    '''

    clip_params = [p.data() for p in model.collect_params().values()]
    if clip is not None:
        norm = nd.array([0.0], ctx)
        for param in clip_params:
            if param.grad is not None:
                norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > clip:
            for param in clip_params:
                if param.grad is not None:
                    param.grad[:] *= clip / norm


def train(train_dataloader, valid_dataloader, test_dataloader, model, loss, trainer, ctx, nepochs, word_vocab,
          char_vocab, label_vocab, clip, init_lr, lr_decay_step, lr_decay_rate, logger):

    # prepare logdir for mxboard
    logdir = '../logs'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    sw = SummaryWriter(logdir=logdir, flush_secs=5, verbose=False)
    global_step = 0
    best_f1_score, best_epoch = 0.0, 0

    for epoch in range(1, nepochs+1):

        for nbatch, (batch_x, batch_char_idx, batch_y) in enumerate(train_dataloader):
            state = model.begin_state(batch_size=batch_x.shape[0], ctx=ctx)
            batch_x = batch_x.as_in_context(ctx)
            batch_char_idx = batch_char_idx.as_in_context(ctx)
            batch_y = batch_y.as_in_context(ctx)

            with autograd.record():
                batch_score, batch_pred, feats, _ = model(batch_char_idx, batch_x, state)
                batch_l = loss(feats, batch_y)

            # clip gradient
            clip_gradients(model, clip, ctx)

            batch_l.backward()
            trainer.step(batch_x.shape[0])
            sw.add_scalar(tag='cross_entroy', value=batch_l.mean().asscalar(), global_step=global_step)
            global_step += 1

            # sampler for log
            if (nbatch + 1) % 300 == 0:
                logger.info("Epoch {0}, n_batch {1}, loss {2}".format(epoch, nbatch + 1, batch_l.mean().asscalar()))

        # evaluate on valid data and test data
        valid_l, valid_P, valid_R, valid_F1 = evaluate(valid_dataloader, model, loss, ctx,
                                                       word_vocab, char_vocab, label_vocab)
        test_l, test_P, test_R, test_F1 = evaluate(test_dataloader, model, loss, ctx,
                                                   word_vocab, char_vocab, label_vocab)
        if test_F1 > best_f1_score:
            best_f1_score = test_F1
            best_epoch = epoch

        # write metrics to the mxboard log
        sw.add_scalar('Loss', {'valid_loss': valid_l}, epoch)
        sw.add_scalar('Loss', {'test_loss': test_l}, epoch)

        sw.add_scalar('Precision', {'valid_P': valid_P}, epoch)
        sw.add_scalar('Precision', {'test_P': test_P}, epoch)

        sw.add_scalar('Recall', {'valid_R': valid_R}, epoch)
        sw.add_scalar('Recall', {'test_R': test_R}, epoch)

        sw.add_scalar('F1', {'valid_F1': valid_F1}, epoch)
        sw.add_scalar('F1', {'test_F1': test_F1}, epoch)

        # sw.add_embedding('Char_Embedding', model.char_embedding_layer.weight.data().asnumpy())

        logger.info('epoch %d, learning_rate %.5f' % (epoch, trainer.learning_rate))
        logger.info('\t valid_loss %.4f, valid_p %.3f, valid_r %.3f, valid_f1 %.3f' %
                    (valid_l, valid_P, valid_R, valid_F1))
        logger.info('\t test_loss %.4f, test_p %.3f, test_r %.3f, test_f1 %.3f' % (test_l, test_P, test_R, test_F1))
        logger.info('\t currently best f1 socre: %.3f on epoch: %d \n' % (best_f1_score, best_epoch))

        # learning rate decay
        if epoch % lr_decay_step == 0:
            trainer.set_learning_rate(init_lr / (1.0+lr_decay_rate*epoch))

        # save params of per epoch
        model_dir = '../models/cnn_bilstm_crf'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_params_path = os.path.join(model_dir, 'cnn_bilstm_crf_model_'+str(epoch)+'.params')
        model.save_parameters(model_params_path)

    sw.export_scalars('../data/scalar_dict.json')
    sw.close()


def evaluate(valid_dataloader, model, loss, ctx, word_vocab, char_vocab, label_vocab):
    valid_l = 0.
    word_seq, true_tag_seq, pred_tag_seq, = [], [], []

    for nbatch, (batch_x, batch_char_idx, batch_y) in enumerate(valid_dataloader):
        state = model.begin_state(batch_size=batch_x.shape[0], ctx=ctx)
        batch_x = batch_x.as_in_context(ctx)
        batch_y = batch_y.as_in_context(ctx)
        batch_char_idx = batch_char_idx.as_in_context(ctx)

        batch_score, batch_pred, feats, _ = model(batch_char_idx, batch_x, state)
        batch_l = loss(feats, batch_y)

        # maybe change
        true_tag_seq.extend(batch_y.asnumpy().astype(np.int32).reshape((-1, )).tolist())
        pred_tag_seq.extend(batch_pred.asnumpy().astype(np.int32).reshape((-1,)).tolist())
        word_seq.extend(batch_x.asnumpy().astype(np.int32).reshape((-1, )).tolist())

        valid_l += batch_l.mean().asscalar()

    assert len(true_tag_seq) == len(pred_tag_seq) == len(word_seq)
    valid_l /= (nbatch + 1)

    word_seq = word_vocab.to_tokens(word_seq)
    pos_seq = ['x']*len(word_seq)
    true_tag_seq = label_vocab.to_tokens(true_tag_seq)
    pred_tag_seq = label_vocab.to_tokens(pred_tag_seq)

    eval_dir = '../data/eval_files'
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    output_file = os.path.join(eval_dir, 'output_file_temp.txt')
    score_file = os.path.join(eval_dir, 'score_file_temp.txt')

    with open(output_file, 'w', encoding='utf-8') as fw:
        for word, pos, true_tag, pred_tag in zip(word_seq, pos_seq, true_tag_seq, pred_tag_seq):
            fw.write(' '.join([word, pos, true_tag, pred_tag]) + '\n')

    acc, p, r, f1 = conll_evaluate(output_file, score_file)

    return valid_l, p, r, f1


