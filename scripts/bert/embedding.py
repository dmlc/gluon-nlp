"""BERT embedding."""
import argparse
import io

import numpy as np
import mxnet as mx
from mxnet.gluon.data import DataLoader
import gluonnlp
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform

try:
    from dataset import BertEmbeddingDataset
except ImportError:
    from .dataset import BertEmbeddingDataset

try:
    unicode
except NameError:
    # Define `unicode` for Python3
    def unicode(s, *_):
        return s


def to_unicode(s):
    return unicode(s, 'utf-8')


__all__ = ['BertEmbedding']


class BertEmbedding(object):
    """
    Encoding from BERT model.

    Parameters
    ----------
    ctx : Context.
        running BertEmbedding on which gpu device id.
    model : str, default bert_12_768_12.
        pre-trained BERT model
    dataset_name : str, default book_corpus_wiki_en_uncased.
        pre-trained model dataset
    max_seq_length : int, default 25
        max length of each sequence
    batch_size : int, default 256
        batch size
    """

    def __init__(self, ctx=mx.cpu(), model='bert_12_768_12',
                 dataset_name='book_corpus_wiki_en_uncased',
                 max_seq_length=25, batch_size=256):
        """
        Encoding from BERT model.

        Parameters
        ----------
        ctx : Context.
            running BertEmbedding on which gpu device id.
        model : str, default bert_12_768_12.
            pre-trained BERT model
        dataset_name : str, default book_corpus_wiki_en_uncased.
            pre-trained model dataset
        max_seq_length : int, default 25
            max length of each sequence
        batch_size : int, default 256
            batch size
        """
        self.ctx = ctx
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.bert, self.vocab = gluonnlp.model.get_model(model,
                                                         dataset_name=dataset_name,
                                                         pretrained=True, ctx=self.ctx,
                                                         use_pooler=False,
                                                         use_decoder=False,
                                                         use_classifier=False)

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
                                         valid_length.astype('float32'))
            for token_id, sequence_output in zip(token_ids.asnumpy(),
                                                 sequence_outputs.asnumpy()):
                batches.append((token_id, sequence_output))
        return self.oov(batches, oov_way)

    def data_loader(self, sentences, shuffle=False):
        tokenizer = BERTTokenizer(self.vocab)
        transform = BERTSentenceTransform(tokenizer=tokenizer,
                                          max_seq_length=self.max_seq_length,
                                          pair=False)
        dataset = BertEmbeddingDataset(sentences, transform)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    def oov(self, batches, oov_way='avg'):
        """
        How to handle oov

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
                    break
                if token_id in (2, 3):
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
    parser.add_argument('--model', type=str, default='bert_12_768_12',
                        help='pre-trained model')
    parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                        help='dataset')
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

    args = parser.parse_args()
    context = mx.gpu(args.gpu) if args.gpu else mx.cpu()
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
        print('Please specify --sentence or --file')

    if result:
        for sent, embeddings in zip(sents, result):
            print('Text: {}'.format(sent))
            _, tokens_embedding = embeddings
            print('Tokens embedding: {}'.format(tokens_embedding))
