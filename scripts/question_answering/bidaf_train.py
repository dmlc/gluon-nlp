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
# pylint:disable=redefined-outer-name

"""Main script to train BiDAF model"""

import copy
import multiprocessing
import os
from time import time

import mxnet as mx
from mxnet import gluon, init, autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

try:
    from bidaf_config import get_args
    from bidaf_evaluate import PerformanceEvaluator
    from data_pipeline import SQuADDataPipeline, SQuADDataLoaderTransformer
    from utils import warm_up_lr, create_output_dir
    from ema import ExponentialMovingAverage
    from bidaf_model import BiDAFModel
except ImportError:
    from .bidaf_config import get_args
    from .bidaf_evaluate import PerformanceEvaluator
    from .data_pipeline import SQuADDataPipeline, SQuADDataLoaderTransformer
    from .utils import warm_up_lr, create_output_dir
    from .ema import ExponentialMovingAverage
    from .bidaf_model import BiDAFModel


def get_context(options):
    """Return context list to work on

    Parameters
    ----------
    options : `Namespace`
        Command arguments

    Returns
    -------
    ctx : list[Context]
        List of contexts
    """
    ctx = []

    if options.gpu is None:
        ctx.append(mx.cpu(0))
        print('Use CPU')
    else:
        indices = options.gpu.split(',')

        for index in indices:
            ctx.append(mx.gpu(int(index)))

    return ctx


def run_training(net, train_dataloader, dev_dataloader, dev_dataset, dev_json, ctx, ema, options):
    """Main function to do training of the network

    Parameters
    ----------
    net : `Block`
        Network to train
    train_dataloader : `DataLoader`
        Initialized training dataloader
    dev_dataloader : `DataLoader`
        Initialized dev dataloader
    dev_dataset : `SQuADQADataset`
        Initialized dev dataset
    dev_json : `dict`
        Original dev JSON data
    ctx: `Context`
        Training context
    ema : `ExponentialMovingAverage`
        Exponential moving average to be used for inference.
    options : `Namespace`
        Training arguments
    """

    hyperparameters = {'learning_rate': options.lr}

    if options.rho:
        hyperparameters['rho'] = options.rho

    trainer = Trainer(net.collect_params(), options.optimizer, hyperparameters, kvstore='device')
    loss_function = SoftmaxCrossEntropyLoss()

    train_start = time()
    avg_loss = mx.nd.zeros((1,), ctx=ctx[0])
    iteration = 1
    max_dev_exact = -1
    max_dev_f1 = -1
    max_iteration = -1
    early_stop_tries = 0

    print('Starting training...')

    for e in range(options.epochs):
        i = 0
        avg_loss *= 0  # Zero average loss of each epoch
        records_per_epoch_count = 0
        e_start = time()

        for i, (r_idx, ctx_words, q_words, ctx_chars, q_chars, start, end) in enumerate(
                train_dataloader):

            records_per_epoch_count += r_idx.shape[0]
            q_words = gluon.utils.split_and_load(q_words, ctx, even_split=False)
            ctx_words = gluon.utils.split_and_load(ctx_words, ctx, even_split=False)
            q_chars = gluon.utils.split_and_load(q_chars, ctx, even_split=False)
            ctx_chars = gluon.utils.split_and_load(ctx_chars, ctx, even_split=False)
            start = gluon.utils.split_and_load(start, ctx, even_split=False)
            end = gluon.utils.split_and_load(end, ctx, even_split=False)

            losses = []

            for qw, cw, qc, cc, s, ee in zip(q_words, ctx_words, q_chars, ctx_chars, start, end):
                with autograd.record():
                    ctx_embedding_state = net.ctx_embedding._contextual_embedding.begin_state(
                        batch_size=r_idx.shape[0], func=mx.ndarray.zeros, ctx=qw.context)

                    modeling_layer_state = net.modeling_layer.begin_state(
                        batch_size=r_idx.shape[0], func=mx.ndarray.zeros, ctx=qw.context)

                    end_index_states = net.output_layer._end_index_lstm.begin_state(
                        batch_size=r_idx.shape[0], func=mx.ndarray.zeros, ctx=qw.context)

                    begin_hat, end_hat = net(qw, cw, qc, cc,
                                             ctx_embedding_state,
                                             modeling_layer_state,
                                             end_index_states)
                    loss = loss_function(begin_hat, s) + loss_function(end_hat, ee)
                    losses.append(loss)
                    # mx.nd.waitall()

            for loss in losses:
                loss.backward()

            # mx.nd.waitall()

            if iteration == 1 and options.use_exponential_moving_average:
                ema.initialize(net.collect_params())

            if options.lr_warmup_steps:
                trainer.set_learning_rate(
                    warm_up_lr(options.lr, iteration, options.lr_warmup_steps))

            execute_trainer_step(net, trainer, ctx, options)

            if options.use_exponential_moving_average:
                ema.update()

            if options.log_interval > 0 and iteration % options.log_interval == 0:
                evaluate_options = copy.deepcopy(options)
                evaluate_options.epochs = iteration
                eval_result = run_evaluate(net, dev_dataloader, dev_dataset, dev_json,
                                           evaluate_options, ema)

                print('Iteration {} evaluation results on dev dataset: {}'.format(iteration,
                                                                                  eval_result))
                if not options.early_stop:
                    continue

                if eval_result['f1'] > max_dev_f1:
                    max_dev_f1 = eval_result['f1']
                    max_dev_exact = eval_result['exact_match']
                    max_iteration = iteration
                    early_stop_tries = 0
                else:
                    early_stop_tries += 1
                    if early_stop_tries < options.early_stop:
                        print('Results decreased for {} times'.format(early_stop_tries))
                    else:
                        print('Results decreased for {} times. Stop training. '
                              'Best results are stored at {} params file. F1={}, EM={}'
                              .format(options.early_stop + 1, max_iteration,
                                      max_dev_f1, max_dev_exact))
                        break

            for l in losses:
                avg_loss += l.mean().as_in_context(avg_loss.context)

            iteration += 1

        mx.nd.waitall()

        avg_loss /= (i * len(ctx))

        # block the call here to get correct Time per epoch
        avg_loss_scalar = avg_loss.asscalar()

        evaluate_options = copy.deepcopy(options)
        evaluate_options.epochs = e
        eval_result = run_evaluate(net, dev_dataloader, dev_dataset, dev_json, evaluate_options,
                                   ema)

        epoch_time = time() - e_start

        print('\tEPOCH {:2}: train loss {:6.4f} | batch {:4} | lr {:5.3f} '
              '| throughtput {:5.3f} of samples/sec | Time per epoch {:5.2f} seconds '
              '| Evaluation result  {}'.format(e, avg_loss_scalar, options.batch_size,
                                               trainer.learning_rate,
                                               records_per_epoch_count / epoch_time,
                                               epoch_time, eval_result))
        if eval_result['f1'] > max_dev_f1:
            max_dev_f1 = eval_result['f1']
            max_dev_exact = eval_result['exact_match']
            save_parameters(net.collect_params(), 'bidaf_model.params', options)
            save_parameters(ema.get_params(), 'bidaf_ema.params', options)

        if options.terminate_training_on_reaching_F1_threshold:
            if eval_result['f1'] >= options.terminate_training_on_reaching_F1_threshold:
                print('Finishing training on {} epoch, because dev F1 score is >= required {}. {}'
                      .format(e, options.terminate_training_on_reaching_F1_threshold, eval_result))
                break

    print('Training time {:6.2f} seconds'.format(time() - train_start))


def get_gradients(model, ctx, options):
    """Get gradients and apply gradient decay to all layers if required.

    Parameters
    ----------
    model : `BiDAFModel`
        Model in training
    ctx : `Context`
        Training context
    options : `Namespace`
        Training options

    Returns
    -------
    gradients : List
        List of gradients
    """
    gradients = []

    for name, parameter in model.collect_params().items():
        if is_fixed_embedding_layer(name) and not options.train_unk_token:
            continue

        grad = parameter.grad(ctx)

        if options.weight_decay:
            if is_fixed_embedding_layer(name):
                grad[0] += options.weight_decay * parameter.data(ctx)[0]
            else:
                grad += options.weight_decay * parameter.data(ctx)

        gradients.append(grad)

    return gradients


def reset_embedding_gradients(model, ctx):
    """Gradients for embedding layer doesn't need to be trained. We train only UNK token of
    embedding if required.

    Parameters
    ----------
    model : `BiDAFModel`
        Model in training
    ctx : `Context`
        Training context
    """
    model.ctx_embedding._word_embedding.weight.grad(ctx=ctx)[1:] = 0


def is_fixed_embedding_layer(name):
    """Check if this is an embedding layer which parameters are supposed to be fixed

    Parameters
    ----------
    name : `str`
        Layer name to check
    """
    return 'predefined_embedding_layer' in name


def execute_trainer_step(net, trainer, ctx, options):
    """Does training step if doesn't need to do gradient clipping or train unknown symbols.

    Parameters
    ----------
    net : `Block`
        Network to train
    trainer : `Trainer`
        Trainer
    ctx: `List[Context]`
        Context list
    options: `SimpleNamespace`
        Training options
    """
    scailing_coeff = len(ctx) * options.batch_size

    if options.clip or options.train_unk_token:
        trainer.allreduce_grads()
        gradients = get_gradients(net, ctx[0], options)

        if options.clip:
            gluon.utils.clip_global_norm(gradients, options.clip)

        if options.train_unk_token:
            reset_embedding_gradients(net, ctx[0])

        if len(ctx) > 1:
            # in multi gpu mode we propagate new gradients to the rest of gpus
            for _, parameter in net.collect_params().items():
                grads = parameter.list_grad()
                source = grads[0]
                destination = grads[1:]

                for dest in destination:
                    source.copyto(dest)

        trainer.update(scailing_coeff)
    else:
        trainer.step(scailing_coeff)


def save_parameters(params, filename, options):
    """Save parameters of the trained model

    Parameters
    ----------
    params : `gluon.ParameterDict`
        Model with trained parameters
    filename : `str`
        Filename of parameters file
    options : `Namespace`
        Saving arguments

    Returns
    -------
    save_path : `str`
        Save path
    """
    if not os.path.exists(options.save_dir):
        os.mkdir(options.save_dir)

    save_path = os.path.join(options.save_dir, filename)
    params.save(save_path)
    return save_path


def run_evaluate(net, dev_dataloader, dev_dataset, dev_json, options, existing_ema=None):
    """Run program in evaluating mode

    Parameters
    ----------
    net : `Block`
        Trained existing network
    dev_dataloader : `DataLoader`
        Dev dataloader
    dev_dataset : `SQuADQADataset`
        Dev dataset
    dev_json : `dict`
        Original JSON dictionary of dev dataset
    existing_ema : `ExponentialMovingAverage`
        Averaged parameters of the network
    options : `Namespace`
        Model evaluation arguments

    Returns
    -------
    result : dict
        Dictionary with exact_match and F1 scores
    """

    if existing_ema is not None:
        params_path = save_parameters(net.collect_params(), 'tmp.params', options)

        for name, params in net.collect_params().items():
            params.set_data(existing_ema.get_param(name))

    evaluator = PerformanceEvaluator(dev_dataloader, dev_dataset, dev_json)
    performance = evaluator.evaluate_performance(net, ctx, options)

    if existing_ema is not None:
        net.collect_params().load(params_path, ctx=ctx)

    return performance


if __name__ == '__main__':
    args = get_args()
    args.batch_size = int(args.batch_size / len(get_context(args)))
    print(args)
    create_output_dir(args.save_dir)

    pipeline = SQuADDataPipeline(args.ctx_max_len, args.q_max_len,
                                 args.ctx_max_len, args.q_max_len,
                                 args.answer_max_len, args.word_max_len,
                                 args.embedding_file_name)
    train_json, dev_json, train_dataset, dev_dataset, word_vocab, char_vocab = \
        pipeline.get_processed_data(use_spacy=False, shrink_word_vocab=True,
                                    filter_train_examples=False)

    ctx = get_context(args)

    ema = ExponentialMovingAverage(args.exponential_moving_average_weight_decay) \
        if args.use_exponential_moving_average else None

    net = BiDAFModel(word_vocab, char_vocab, args, prefix='bidaf')
    net.initialize(init.Xavier(), ctx=ctx)
    net.hybridize(static_alloc=True)

    train_dataloader = DataLoader(train_dataset.transform(SQuADDataLoaderTransformer()),
                                  batch_size=len(ctx) * args.batch_size, shuffle=True,
                                  last_batch='rollover', num_workers=multiprocessing.cpu_count(),
                                  pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset.transform(SQuADDataLoaderTransformer()),
                                batch_size=len(ctx) * args.batch_size, shuffle=False,
                                last_batch='keep', num_workers=multiprocessing.cpu_count(),
                                pin_memory=True)

    if args.train or not args.evaluate:
        print('Running in training mode')
        run_training(net, train_dataloader, dev_dataloader, dev_dataset, dev_json, ctx, ema,
                     options=args)

    if args.evaluate:
        print('Running in evaluation mode')
        result = run_evaluate(net, dev_dataloader, dev_dataset, dev_json, args)
        print('Evaluation results on dev dataset: {}'.format(result))
