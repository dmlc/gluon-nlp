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

# pylint: disable=
r"""
This file contains the train code.
"""
import json
import time
import logging

from mxnet import autograd, gluon, nd
from tqdm import tqdm

from config import CTX, opt
from data_loader import DataLoader
from evaluate import evaluate

from model import MySoftmaxCrossEntropy, QANet
from util import ExponentialMovingAverage, load_emb_mat, warm_up_lr

accum_avg_train_ce = []
batch_train_ce = []
dev_f1 = []
dev_em = []
global_step = 0
logging.basicConfig(level=logging.DEBUG)


def train(model, data_loader, trainer, loss_function, ema=None):
    r"""
    Train and save.
    """
    for e in tqdm(range(opt.epochs)):
        logging.debug('Begin %d/%d epoch...', e + 1, opt.epochs)
        train_one_epoch(model, data_loader, trainer, loss_function, ema)

        # save model after train one epoch
        model.save_parameters(opt.prefix_model +
                              time.asctime(time.localtime(time.time())))
        trainer.save_states(opt.prefix_trainer +
                            time.asctime(time.localtime(time.time())))
        with open('accum_train_cross_entropy', 'w') as f:
            f.write(json.dumps(accum_avg_train_ce))
        with open('batch_train_cross_entorpy', 'w') as f:
            f.write(json.dumps(batch_train_ce))


def train_one_epoch(model, data_loader, trainer, loss_function, ema=None):
    r"""
    One train loop.
    """
    total_batchs = data_loader.total_batchs
    total_loss = 0
    step = 0
    global global_step
    for batch_data in data_loader.next_batch():
        step += 1
        global_step += 1
        # add evaluate per evaluate_interval batchs
        if global_step % opt.evaluate_interval == 0:
            logging.debug('\nglobal_step == %d', global_step)
            logging.info('evaluating dev dataset...')
            f1_score, em_score = evaluate(model, dataset_type='dev', ema=ema)
            logging.info('dev f1:' + str(f1_score) + 'em:' + str(em_score))
            dev_f1.append([global_step, f1_score])
            dev_em.append([global_step, em_score])

        context = nd.array([x[0] for x in batch_data])
        query = nd.array([x[1] for x in batch_data])
        context_char = nd.array([x[2] for x in batch_data])
        query_char = nd.array([x[3] for x in batch_data])
        begin = nd.array([x[4] for x in batch_data])
        end = nd.array([x[5] for x in batch_data])
        batch_sizes = context.shape[0]
        context = gluon.utils.split_and_load(
            data=context,
            ctx_list=CTX
        )

        query = gluon.utils.split_and_load(
            data=query,
            ctx_list=CTX
        )
        context_char = gluon.utils.split_and_load(
            data=context_char,
            ctx_list=CTX
        )
        query_char = gluon.utils.split_and_load(
            data=query_char,
            ctx_list=CTX
        )
        begin = gluon.utils.split_and_load(
            data=begin,
            ctx_list=CTX
        )
        end = gluon.utils.split_and_load(
            data=end,
            ctx_list=CTX
        )

        with autograd.record():
            different_ctx_loss = [loss_function(*model(c, q, cc, qc, b, e))
                                  for c, q, cc, qc, b, e in
                                  zip(context, query, context_char, query_char, begin, end)]

            for loss in different_ctx_loss:
                loss.backward()
        if global_step == 1:
            for name, param in model.collect_params().items():
                ema.add(name, param.data(CTX[0]))
        trainer.set_learning_rate(warm_up_lr(global_step))
        trainer.allreduce_grads()
        reset_embedding_grad(model)
        tmp = []
        for name, paramater in model.collect_params().items():
            grad = paramater.grad(context[0].context)
            if name == 'qanet0_embedding0_weight':
                grad[0:2] += opt.weight_decay * \
                    paramater.data(context[0].context)[0:2]
            else:
                grad += opt.weight_decay * paramater.data(context[0].context)
            tmp.append(grad)
        gluon.utils.clip_global_norm(tmp, opt.clip_gradient)
        reset_embedding_grad(model)
        trainer.update(batch_sizes, ignore_stale_grad=True)
        for name, param in model.collect_params().items():
            ema(name, param.data(CTX[0]))

        batch_loss = .0
        for loss in different_ctx_loss:
            batch_loss += loss.mean().asscalar()
        batch_loss /= len(different_ctx_loss)
        total_loss += batch_loss

        batch_train_ce.append([global_step, batch_loss])
        accum_avg_train_ce.append([global_step, total_loss / step])

        print('batch %d/%d, total_loss %.2f, batch_loss %.2f' %
              (step, total_batchs, total_loss / step, batch_loss), end='\r', flush=True)
        nd.waitall()


def initial_model_parameters(model):
    r"""
    Initial the word embedding layer.
    """
    model.collect_params().initialize(ctx=CTX)
    # initial embedding parameters with glove
    word_embedding = nd.array(load_emb_mat(opt.word_emb_file_name))

    model.word_emb[0].weight.set_data(word_embedding)


def reset_embedding_grad(model):
    r"""
    Reset the grad about word embedding layer.
    """
    for ctx in CTX:
        model.word_emb[0].weight.grad(ctx=ctx)[2:] = 0


def main():
    r"""
    Main function.
    """
    model = QANet()
    # initial parameters
    logging.info('Initial paramters...')
    if opt.load_trained_model:
        model.load_parameters(opt.trained_model_name, ctx=CTX)
    else:
        logging.info('Initial model parameters...')
        initial_model_parameters(model)
    print(model)
    if opt.is_train:
        loss_function = MySoftmaxCrossEntropy()

        ema = ExponentialMovingAverage(decay=opt.ema_decay)

        # initial trainer
        trainer = gluon.Trainer(
            model.collect_params(),
            'adam',
            {
                'learning_rate': opt.init_learning_rate,
                'beta1': opt.beta1,
                'beta2': opt.beta2,
                'epsilon': opt.epsilon
            }
        )

        if opt.load_trained_model:
            trainer.load_states(opt.trained_trainer_name)

        # initial dataloader
        train_data_loader = DataLoader(
            batch_size=opt.train_batch_size, dev_set=False)

        # train
        logging.info('Train')
        train(model, train_data_loader, trainer, loss_function, ema)
    else:
        logging.info('Evaluating dev set...')
        f1_score, em_score = evaluate(model, dataset_type='dev', ema=None)
        logging.debug(
            'The dev dataset F1 is:%.5f, and EM is: %.5f', f1_score, em_score)


if __name__ == '__main__':
    main()
