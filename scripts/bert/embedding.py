"""BERT embedding."""
import argparse
import io
import logging
import os

import numpy as np
import mxnet as mx

from mxnet.gluon.data import DataLoader

import gluonnlp
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from gluonnlp.base import get_home_dir

try:
    from data.embedding import BertEmbeddingDataset
except ImportError:
    from .data.embedding import BertEmbeddingDataset

try:
    unicode
except NameError:
    # Define `unicode` for Python3
    def unicode(s, *_):
        return s


def to_unicode(s):
    return unicode(s, 'utf-8')


__all__ = ['BertEmbedding']


logger = logging.getLogger(__name__)


class BertEmbedding(object):
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
    root : str, default '$MXNET_HOME/models' with MXNET_HOME defaults to '~/.mxnet'
        Location for keeping the model parameters.
    """
    def __init__(self, ctx=mx.cpu(), dtype='float32', model='bert_12_768_12',
                 dataset_name='book_corpus_wiki_en_uncased', params_path=None,
                 max_seq_length=25, batch_size=256,
                 root=os.path.join(get_home_dir(), 'models')):
        self.ctx = ctx
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        # Don't download the pretrained models if we have a parameter path
        self.bert, self.vocab = gluonnlp.model.get_model(model,
                                                         dataset_name=self.dataset_name,
                                                         pretrained=params_path is None,
                                                         ctx=self.ctx,
                                                         use_pooler=False,
                                                         use_decoder=False,
                                                         use_classifier=False,
                                                         root=root)
        self.bert.cast(self.dtype)

        if params_path:
            logger.info('Loading params from %s', params_path)
            self.bert.load_parameters(params_path, ctx=ctx, ignore_extra=True)

        lower = 'uncased' in self.dataset_name
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
        batches : List[(tokens_id,
                        sequence_outputs,
                        pooled_output].
            batch   token_ids (max_seq_length, ),
                    sequence_outputs (max_seq_length, dim, ),
                    pooled_output (dim, )
        oov_way : str
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        sentences = []
        for token_ids, sequence_outputs in batches:
            tokens = []
            tensors = []
            oov_len = 1
            for token_id, sequence_output in zip(token_ids, sequence_outputs):
                if token_id == 1:
                    # [PAD] token, sequence is finished.
                    break
                if token_id in (2, 3):
                    # [CLS], [SEP]
                    continue
                token = self.vocab.idx_to_token[token_id]
                if token.startswith('##'):
                    token = token[2:]
                    tokens[-1] += token
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
                        help='dataset')
    parser.add_argument('--params_path', type=str, default=None,
                        help='path to a params file to load instead of the pretrained model.')
    parser.add_argument('--max_seq_length', type=int, default=25,
                        help='max length of each sequence')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--oov_way', type=str, default='avg',
                        help='how to handle oov\n'
                             'avg: average all oov embeddings to represent the original token\n'
                             'sum: sum all oov embeddings to represent the original token\n'
                             'last: use last oov embeddings to represent the original token\n')
    parser.add_argument('--sentences', type=to_unicode, nargs='+', default=None,
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
                                   max_seq_length=args.max_seq_length, batch_size=args.batch_size)
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
        for sent, embeddings in zip(sents, result):
            print('Text: {}'.format(sent))
            _, tokens_embedding = embeddings
            print('Tokens embedding: {}'.format(tokens_embedding))
