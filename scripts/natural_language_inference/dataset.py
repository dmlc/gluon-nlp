"""
nlidataset.py

Part of NLI script in gluon-nlp.
Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>
"""

import os
import logging
from mxnet import gluon
import gluonnlp as nlp
import gluonnlp.data.batchify as btf

logger = logging.getLogger('nli')
LABEL_TO_IDX = {'neutral': 0, 'contradiction': 1, 'entailment': 2}

def read_dataset(args, dataset):
    path = os.path.join(vars(args)[dataset])
    logger.info('reading data from {}'.format(path))
    examples = [line.strip().split('\t') for line in open(path)]
    if args.max_num_examples > 0:
        examples = examples[:args.max_num_examples]
    # NOTE: assume data has been tokenized
    dataset = gluon.data.SimpleDataset([(e[0], e[1], LABEL_TO_IDX[e[2]]) for e in examples])
    dataset = dataset.transform(lambda s1, s2, label: (
                                ['NULL'] + s1.lower().split(),
                                ['NULL'] + s2.lower().split(), label),
                                lazy=False)
    return dataset

def build_vocab(dataset):
    counter = nlp.data.count_tokens([w for e in dataset for s in e[:2] for w in s],
                                    to_lower=True)
    vocab = nlp.Vocab(counter)
    return vocab

def prepare_data_loader(args, dataset, vocab, test=False):
    """
    Read data `dataset` and prepare data loader for it.
    Illegal examples are removed.
    """
    # Preprocess
    dataset = dataset.transform(lambda s1, s2, label: (vocab(s1), vocab(s2), label),
                                lazy=False)

    # Batching
    batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(), btf.Stack(dtype='int32'))
    data_lengths = [max(len(d[0]), len(d[1])) for d in dataset]
    batch_sampler = nlp.data.FixedBucketSampler(lengths=data_lengths,
                                       batch_size=args.batch_size,
                                       shuffle=(not test))
    data_loader = gluon.data.DataLoader(dataset=dataset,
                                   batch_sampler=batch_sampler,
                                   batchify_fn=batchify_fn)
    return data_loader
