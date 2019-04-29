#!/usr/bin/env python
# coding: utf-8
import argparse
import logging

import numpy as np
import mxnet as mx
import random

import gluonnlp as nlp

from ner_common import *
from ner_data import BERTTaggingDataset, convert_arrays_to_text
from ner_model import BERTTagger, attach_prediction

# TODO: currently, our evaluation is dependent on this package. figure out whether to take actual dependency on it.
import seqeval.metrics


def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(
        description="Train a BERT-based named entity recognition model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data file paths
    arg_parser.add_argument("--train-path", type=str, required=True,
                            help="Path to the training data file")
    arg_parser.add_argument("--dev-path", type=str, required=True,
                            help="Path to the development data file")
    arg_parser.add_argument("--test-path", type=str, required=True,
                            help="Path to the test data file")

    arg_parser.add_argument("--save-checkpoint-prefix", type=str, required=False, default=None,
                            help="Prefix of model checkpoint file")

    # bert options
    arg_parser.add_argument("--bert-model", type=str, default="bert_12_768_12",
                            help="Name of the BERT model")
    arg_parser.add_argument("--cased", type=str2bool, default=True,
                            help="Path to the development data file")
    arg_parser.add_argument("--dropout-prob", type=float, default=0.1,
                            help="Dropout probability for the last layer")

    # optimization parameters
    arg_parser.add_argument("--seed", type=int, default=13531,
                            help='Random number seed.')
    arg_parser.add_argument("--seq-len", type=int, default=180,
                            help="The length of the sequence input to BERT."
                                 " An exception will raised if this is not large enough.")
    arg_parser.add_argument("--gpu", type=int,
                            help='Number (index) of GPU to run on, e.g. 0.  If not specified, uses CPU.')
    arg_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    arg_parser.add_argument("--num-epochs", type=int, default=4, help="Number of epochs to train")
    arg_parser.add_argument("--optimizer", type=str, default="bertadam", help="Optimization algorithm to use")
    arg_parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for optimization")
    arg_parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduling")
    args = arg_parser.parse_args()
    return args


def main(config):
    # provide random seed for every RNGs we use
    np.random.seed(config.seed)
    random.seed(config.seed)
    mx.random.seed(config.seed)

    ctx = get_context(config.gpu)

    logging.info("Loading BERT model...")
    bert_model, text_vocab = get_bert_model(config.bert_model, config.cased, ctx, config.dropout_prob)

    dataset = BERTTaggingDataset(text_vocab, config.train_path, config.dev_path, config.test_path,
                                 config.seq_len, config.cased)

    train_data_loader = dataset.get_train_data_loader(config.batch_size)
    dev_data_loader = dataset.get_dev_data_loader(config.batch_size)
    test_data_loader = dataset.get_test_data_loader(config.batch_size)

    net = BERTTagger(bert_model, dataset.num_tag_types, config.dropout_prob)
    net.tag_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    net.hybridize(static_alloc=True)

    loss_function = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    loss_function.hybridize(static_alloc=True)

    # step size adaptation, adopted from: https://github.com/dmlc/gluon-nlp/blob/87d36e3cc7c615f93732d01048cf7ce3b3b09eb7/scripts/bert/finetune_classifier.py#L348-L351
    step_size = config.batch_size
    num_train_steps = int(len(dataset.train_inputs) / step_size * config.num_epochs)
    num_warmup_steps = int(num_train_steps * config.warmup_ratio)

    optimizer_params = {'learning_rate': config.learning_rate}
    try:
        trainer = mx.gluon.Trainer(net.collect_params(), config.optimizer, optimizer_params)
    except ValueError as e:
        print(e)
        logging.warning('AdamW optimizer is not found. Please consider upgrading to '
                        'mxnet>=1.5.0. Now the original Adam optimizer is used instead.')
        trainer = mx.gluon.Trainer(net.collect_params(), 'adam', optimizer_params)

    # collect differentiable parameters
    logging.info("Collect params...")
    # do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    params = [p for p in net.collect_params().values() if p.grad_req != 'null']

    if config.save_checkpoint_prefix is not None:
        logging.info("dumping metadata...")
        dump_metadata(config, tag_vocab=dataset.tag_vocab)

    def train(data_loader, start_step_num):
        step_num = start_step_num
        logging.info("current starting step num: {}".format(step_num))
        for batch_id, (text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag, out) in \
                enumerate(attach_prediction(data_loader, net, ctx, is_train=True)):
            logging.info("training on batch index: {}/{}".format(batch_id, len(data_loader)))

            # step size adjustments
            step_num += 1
            if step_num < num_warmup_steps:
                new_lr = config.learning_rate * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * config.learning_rate / (
                        num_train_steps - num_warmup_steps)
                new_lr = config.learning_rate - offset
            trainer.set_learning_rate(new_lr)

            with mx.autograd.record():
                loss_value = loss_function(out, tag_ids, flag_nonnull_tag.expand_dims(axis=2)).mean()

            loss_value.backward()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer.step(1)

            pred_tags = out.argmax(axis=-1)
            logging.info("loss_value: {:.6f}".format(loss_value.asscalar()))

            num_tag_preds = flag_nonnull_tag.sum().asscalar()
            logging.info(
                "accuracy: {:.6f}".format(((pred_tags == tag_ids) * flag_nonnull_tag).sum().asscalar() / num_tag_preds))
        return step_num

    def evaluate(data_loader):
        predictions = []

        for batch_id, (text_ids, token_types, valid_length, tag_ids, flag_nonnull_tag, out) in \
                enumerate(attach_prediction(data_loader, net, ctx, is_train=False)):
            logging.info("evaluating on batch index: {}/{}".format(batch_id, len(data_loader)))

            # convert results to numpy arrays for easier access
            np_text_ids = text_ids.astype('int32').asnumpy()
            np_pred_tags = out.argmax(axis=-1).asnumpy()
            np_valid_length = valid_length.astype('int32').asnumpy()
            np_true_tags = tag_ids.asnumpy()

            predictions += convert_arrays_to_text(text_vocab, dataset.tag_vocab, np_text_ids,
                                                  np_true_tags, np_pred_tags, np_valid_length)

        all_true_tags = [[entry.true_tag for entry in entries] for entries in predictions]
        all_pred_tags = [[entry.pred_tag for entry in entries] for entries in predictions]
        seqeval_f1 = seqeval.metrics.f1_score(all_true_tags, all_pred_tags)
        return seqeval_f1

    best_dev_f1 = 0.0
    best_epoch = -1

    last_epoch_step_num = 0
    for epoch_index in range(config.num_epochs):
        last_epoch_step_num = train(train_data_loader, last_epoch_step_num)
        train_f1 = evaluate(train_data_loader)
        logging.info("train f1: {:.3f}".format(train_f1))
        dev_f1 = evaluate(dev_data_loader)
        logging.info("dev f1: {:.3f}, previous best dev f1: {:.3f}".format(dev_f1, best_dev_f1))
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch_index
            logging.info("update the best dev f1 to be: {:.3f}".format(best_dev_f1))
            test_f1 = evaluate(test_data_loader)
            logging.info("test f1: {:.3f}".format(test_f1))

            # save params
            params_file = config.save_checkpoint_prefix + "_{:03d}.params".format(epoch_index)
            logging.info("saving current checkpoint to: {}".format(params_file))
            net.save_parameters(params_file)

        logging.info("current best epoch: {:d}".format(best_epoch))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    main(parse_args())
