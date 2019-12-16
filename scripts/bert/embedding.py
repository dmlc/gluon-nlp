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
"""BERT embedding."""

import argparse
import io
import logging
import os

import numpy as np
import mxnet as mx

from mxnet.gluon.data import DataLoader

import gluonnlp
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform, BERTSPTokenizer
from gluonnlp.base import get_home_dir

try:
    from data.embedding import BertEmbeddingDataset
except ImportError:
    from .data.embedding import BertEmbeddingDataset


__all__ = ['BertEmbedding']


logger = logging.getLogger(__name__)


class BertEmbedding:
    """
    Encoding from BERT model.

    Parameters
    ----------
    ctx : Context.
        running BertEmbedding on which gpu device id.
    dtype: str
        data type to use for the model.
    model : str, default bert_12_768_12.
        pre-trained BERT model
    dataset_name : str, default book_corpus_wiki_en_uncased.
        pre-trained model dataset
    params_path: str, default None
        path to a parameters file to load instead of the pretrained model.
    max_seq_length : int, default 25
        max length of each sequence
    batch_size : int, default 256
        batch size
    sentencepiece : str, default None
        Path to the sentencepiece .model file for both tokenization and vocab
    root : str, default '$MXNET_HOME/models' with MXNET_HOME defaults to '~/.mxnet'
        Location for keeping the model parameters.
    """
    def __init__(self, ctx=mx.cpu(), dtype='float32', model='bert_12_768_12',
                 dataset_name='book_corpus_wiki_en_uncased', params_path=None,
                 max_seq_length=25, batch_size=256, sentencepiece=None,
                 root=os.path.join(get_home_dir(), 'models')):
        self.ctx = ctx
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        # use sentencepiece vocab and a checkpoint
        # we need to set dataset_name to None, otherwise it uses the downloaded vocab
        if params_path and sentencepiece:
            dataset_name = None
        else:
            dataset_name = self.dataset_name
        if sentencepiece:
            vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(sentencepiece)
        else:
            vocab = None
        self.bert, self.vocab = gluonnlp.model.get_model(model,
                                                         dataset_name=dataset_name,
                                                         pretrained=params_path is None,
                                                         ctx=self.ctx,
                                                         use_pooler=False,
                                                         use_decoder=False,
                                                         use_classifier=False,
                                                         root=root, vocab=vocab)

        self.bert.cast(self.dtype)
        if params_path:
            logger.info('Loading params from %s', params_path)
            self.bert.load_parameters(params_path, ctx=ctx, ignore_extra=True, cast_dtype=True)

        lower = 'uncased' in self.dataset_name
        if sentencepiece:
            self.tokenizer = BERTSPTokenizer(sentencepiece, self.vocab, lower=lower)
        else:
            self.tokenizer = BERTTokenizer(self.vocab, lower=lower)
        self.transform = BERTSentenceTransform(tokenizer=self.tokenizer,
                                               max_seq_length=self.max_seq_length,
                                               pair=False)

    def __call__(self, sentences, oov_way='avg'):
        return self.embedding(sentences, oov_way='avg')

    def embedding(self, sentences, oov_way='avg'):
        """
        Get tokens, tokens embedding

        Parameters
        ----------
        sentences : List[str]
            sentences for encoding.
        oov_way : str, default avg.
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        data_iter = self.data_loader(sentences=sentences)
        batches = []
        for token_ids, valid_length, token_types in data_iter:
            token_ids = token_ids.as_in_context(self.ctx)
            valid_length = valid_length.as_in_context(self.ctx)
            token_types = token_types.as_in_context(self.ctx)
            sequence_outputs = self.bert(token_ids, token_types,
                                         valid_length.astype(self.dtype))
            for token_id, sequence_output in zip(token_ids.asnumpy(),
                                                 sequence_outputs.asnumpy()):
                batches.append((token_id, sequence_output))
        return self.oov(batches, oov_way)

    def data_loader(self, sentences, shuffle=False):
        """Load, tokenize and prepare the input sentences."""
        dataset = BertEmbeddingDataset(sentences, self.transform)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    def oov(self, batches, oov_way='avg'):
        """
        How to handle oov. Also filter out [CLS], [SEP] tokens.

        Parameters
        ----------
        batches : List[(tokens_id, sequence_outputs)].
            batch   token_ids shape is (max_seq_length,),
                    sequence_outputs shape is (max_seq_length, dim)
        oov_way : str
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        sentences = []
        padding_idx, cls_idx, sep_idx = None, None, None
        if self.vocab.padding_token:
            padding_idx = self.vocab[self.vocab.padding_token]
        if self.vocab.cls_token:
            cls_idx = self.vocab[self.vocab.cls_token]
        if self.vocab.sep_token:
            sep_idx = self.vocab[self.vocab.sep_token]
        for token_ids, sequence_outputs in batches:
            tokens = []
            tensors = []
            oov_len = 1
            for token_id, sequence_output in zip(token_ids, sequence_outputs):
                # [PAD] token, sequence is finished.
                if padding_idx and token_id == padding_idx:
                    break
                # [CLS], [SEP]
                if cls_idx and token_id == cls_idx:
                    continue
                if sep_idx and token_id == sep_idx:
                    continue
                token = self.vocab.idx_to_token[token_id]
                if not self.tokenizer.is_first_subword(token):
                    tokens.append(token)
                    if oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if oov_way == 'avg':
                        oov_len += 1
                else:  # iv, avg last oov
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)
            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                tensors[-1] /= oov_len
            sentences.append((tokens, tensors))
        return sentences


if __name__ == '__main__':
    np.set_printoptions(threshold=5)
    parser = argparse.ArgumentParser(description='Get embeddings from BERT',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--gpu', type=int, default=None,
                        help='id of the gpu to use. Set it to empty means to use cpu.')
    parser.add_argument('--dtype', type=str, default='float32', help='data dtype')
    parser.add_argument('--model', type=str, default='bert_12_768_12',
                        help='pre-trained model')
    parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                        help='name of the dataset used for pre-training')
    parser.add_argument('--params_path', type=str, default=None,
                        help='path to a params file to load instead of the pretrained model.')
    parser.add_argument('--sentencepiece', type=str, default=None,
                        help='Path to the sentencepiece .model file for tokenization and vocab.')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='max length of each sequence')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--oov_way', type=str, default='avg',
                        help='how to handle subword embeddings\n'
                             'avg: average all subword embeddings to represent the original token\n'
                             'sum: sum all subword embeddings to represent the original token\n'
                             'last: use last subword embeddings to represent the original token\n')
    parser.add_argument('--sentences', type=str, nargs='+', default=None,
                        help='sentence for encoding')
    parser.add_argument('--file', type=str, default=None,
                        help='file for encoding')
    parser.add_argument('--verbose', action='store_true', help='verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(level)
    logging.info(args)

    if args.gpu is not None:
        context = mx.gpu(args.gpu)
    else:
        context = mx.cpu()
    bert_embedding = BertEmbedding(ctx=context, model=args.model, dataset_name=args.dataset_name,
                                   max_seq_length=args.max_seq_length, batch_size=args.batch_size,
                                   params_path=args.params_path, sentencepiece=args.sentencepiece)
    result = []
    sents = []
    if args.sentences:
        sents = args.sentences
        result = bert_embedding(sents, oov_way=args.oov_way)
    elif args.file:
        with io.open(args.file, 'r', encoding='utf8') as in_file:
            for line in in_file:
                sents.append(line.strip())
        result = bert_embedding(sents, oov_way=args.oov_way)
    else:
        logger.error('Please specify --sentence or --file')

    if result:
        for _, embeddings in zip(sents, result):
            sent, tokens_embedding = embeddings
            print('Text: {}'.format(' '.join(sent)))
            print('Tokens embedding: {}'.format(tokens_embedding))
