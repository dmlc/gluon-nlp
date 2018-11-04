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

from collections import Counter
import numpy as np

from gluonnlp.model.biaffine.common.k_means import KMeans
from .savable import Savable


class ConllWord(object):
    def __init__(self, id, form, lemma=None, cpos=None, pos=None, feats=None, head=None, relation=None, phead=None,
                 pdeprel=None):
        """
        CoNLL format template, see http://anthology.aclweb.org/W/W06/W06-2920.pdf
        :param id: Token counter, starting at 1 for each new sentence.
        :param form: Word form or punctuation symbol.
        :param lemma: Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
        :param cpos: Coarse-grained part-of-speech tag, where the tagset depends on the treebank.
        :param pos: Fine-grained part-of-speech tag, where the tagset depends on the treebank.
        :param feats: Unordered set of syntactic and/or morphological features (depending on the particular treebank), or an underscore if not available.
        :param head: Head of the current token, which is either a value of ID, or zero (’0’) if the token links to the virtual root node of the sentence.
        :param relation: Dependency relation to the HEAD.
        :param phead: Projective head of current token, which is either a value of ID or zero (’0’), or an underscore if not available.
        :param pdeprel: Dependency relation to the PHEAD, or an underscore if not available.
        """
        self.id = id
        self.form = form
        self.cpos = cpos
        self.pos = pos
        self.head = head
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.phead = phead
        self.pdeprel = pdeprel

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats, str(self.head), self.relation,
                  self.phead, self.pdeprel]
        return '\t'.join(['_' if v is None else v for v in values])


class ConllSentence(object):
    def __init__(self, words: list):
        super().__init__()
        self.words = words

    def __str__(self):
        return '\n'.join([word.__str__() for word in self.words])

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        return self.words[index]

    def __iter__(self):
        return (line for line in self.words)


class ParserVocabulary(Savable):
    """
    Vocabulary, holds word, tag and relation along with their id
    Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser with some modifications
    """
    PAD, ROOT, UNK = 0, 1, 2
    """Padding, Root, Unknown"""

    def __init__(self, input_file, pret_file=None, min_occur_count=2):
        """
        Load from conll file
        :param input_file: conll file
        :param pret_file: word vector file
        :param min_occur_count: threshold of word frequency
        """
        super().__init__()
        word_counter = Counter()
        tag_set = set()
        rel_set = set()

        with open(input_file) as f:
            for line in f:
                info = line.strip().split()
                if info:
                    if len(info) == 10:
                        arc_offset = 6
                        rel_offset = 7
                    elif len(info) == 8:
                        arc_offset = 5
                        rel_offset = 6
                    word, tag, head, rel = info[1].lower(), info[3], int(info[arc_offset]), info[rel_offset]
                    word_counter[word] += 1
                    tag_set.add(tag)
                    if rel != 'root':
                        rel_set.add(rel)

        self._id2word = ['<pad>', '<root>', '<unk>']
        self._id2tag = ['<pad>', '<root>', '<unk>']
        self._id2rel = ['<pad>', 'root']
        reverse = lambda x: dict(list(zip(x, list(range(len(x))))))
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)

        self._pret_file = pret_file
        self._words_in_train_data = len(self._id2word)
        # print('#words in training set:', self._words_in_train_data)
        if pret_file:
            self._add_pret_words(pret_file)
        self._id2tag += list(tag_set)
        self._id2rel += list(rel_set)

        self._word2id = reverse(self._id2word)
        self._tag2id = reverse(self._id2tag)
        self._rel2id = reverse(self._id2rel)
        # print("Vocab info: #words %d, #tags %d #rels %d" % (self.vocab_size, self.tag_size, self.rel_size))

    def log_info(self, logger):
        """
        Print statistical information via the provided logger
        :param logger: logger created using logging.getLogger()
        """
        logger.info('#words in training set: %d' % self._words_in_train_data)
        logger.info("Vocab info: #words %d, #tags %d #rels %d" % (self.vocab_size, self.tag_size, self.rel_size))

    def _add_pret_words(self, pret_file):
        """
        Read pre-trained embedding file for extending vocabulary
        :param pret_file: path to pre-trained embedding file
        """
        words_in_train_data = set(self._id2word)
        with open(pret_file) as f:
            for line in f:
                line = line.strip().split()
                if line:
                    word = line[0]
                    if word not in words_in_train_data:
                        self._id2word.append(word)
                        # print 'Total words:', len(self._id2word)

    def has_pret_embs(self):
        """
        Whether this vocabulary contains words from pre-trained embeddings
        :return:
        """
        return self._pret_file is not None

    def get_pret_embs(self, word_dims=None):
        """
        Read pre-trained embedding file
        :param word_dims: vector size. Use `None` for auto-infer
        :return: T x C numpy NDArray
        """
        assert (self._pret_file is not None), "No pretrained file provided."
        embs = [[]] * len(self._id2word)
        train = True
        try:
            with open(self._pret_file) as f:
                dim = None
                for line in f:
                    line = line.strip().split()
                    if len(line) > 2:
                        if dim is None:
                            dim = len(line)
                        else:
                            if len(line) != dim:
                                continue
                        word, data = line[0], line[1:]
                        embs[self._word2id[word]] = data
        except FileNotFoundError:
            train = False
        if word_dims is None:
            word_dims = len(data)
        for idx, emb in enumerate(embs):
            if not emb:
                embs[idx] = np.zeros(word_dims)
        pret_embs = np.array(embs, dtype=np.float32)
        return pret_embs / np.std(pret_embs) if train else pret_embs

    def get_word_embs(self, word_dims):
        """
        Get randomly initialized embeddings when pre-trained embeddings are used, otherwise zero vectors
        :param word_dims: word vector size
        :return: T x C numpy NDArray
        """
        if self._pret_file is not None:
            return np.random.randn(self.words_in_train, word_dims).astype(np.float32)
        return np.zeros((self.words_in_train, word_dims), dtype=np.float32)

    def get_tag_embs(self, tag_dims):
        """
        Randomly initialize embeddings for tag
        :param tag_dims: tag vector size
        :return: numpy NDArray
        """
        return np.random.randn(self.tag_size, tag_dims).astype(np.float32)

    def word2id(self, xs):
        """
        Map word(s) to its id(s)
        :param xs: word or a list of words
        :return: id or a list of ids
        """
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        """
        Map id(s) to word(s)
        :param xs: id or a list of ids
        :return: word or a list of words
        """
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def rel2id(self, xs):
        """
        Map relation(s) to id(s)
        :param xs: relation
        :return: id
        """
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        return self._rel2id[xs]

    def id2rel(self, xs):
        """
        Map id(s) to relation(s)
        :param xs:  id
        :return: relation
        """
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def tag2id(self, xs):
        """
        Map tag(s) to id(s)
        :param xs: tag
        :return: id
        """
        if isinstance(xs, list):
            return [self._tag2id.get(x, self.UNK) for x in xs]
        return self._tag2id.get(xs, self.UNK)

    @property
    def words_in_train(self):
        """
        #words in training set
        :return:
        """
        return self._words_in_train_data

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def rel_size(self):
        return len(self._id2rel)


class DataLoader(object):
    """
    Load CoNLL data
    Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser with some modifications
    """

    def __init__(self, input_file, n_bkts, vocab):
        """
        Begin loading
        :param input_file: CoNLL file
        :param n_bkts: number of buckets
        :param vocab: vocabulary object
        """
        self.vocab = vocab
        sents = []
        sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0, ParserVocabulary.ROOT]]
        with open(input_file) as f:
            for line in f:
                info = line.strip().split()
                if info:
                    arc_offset = 5
                    rel_offset = 6
                    if len(info) == 10:
                        arc_offset = 6
                        rel_offset = 7
                    # else:
                    #     raise RuntimeError('Illegal line: %s' % line)
                    assert info[rel_offset] in vocab._rel2id, 'Relation OOV: %s' % line
                    word, tag, head, rel = vocab.word2id(info[1].lower()), vocab.tag2id(info[3]), int(
                        info[arc_offset]), vocab.rel2id(info[rel_offset])
                    sent.append([word, tag, head, rel])
                else:
                    sents.append(sent)
                    sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0, ParserVocabulary.ROOT]]
            if len(sent) > 1:  # last sent in file without '\n'
                sents.append(sent)

        self.samples = len(sents)
        len_counter = Counter()
        for sent in sents:
            len_counter[len(sent)] += 1
        self._bucket_lengths = KMeans(n_bkts, len_counter).splits
        self._buckets = [[] for i in range(n_bkts)]
        """bkt_idx x length x sent_idx x 4"""
        len2bkt = {}
        prev_length = -1
        for bkt_idx, length in enumerate(self._bucket_lengths):
            len2bkt.update(list(zip(list(range(prev_length + 1, length + 1)), [bkt_idx] * (length - prev_length))))
            prev_length = length

        self._record = []
        for sent in sents:
            bkt_idx = len2bkt[len(sent)]
            idx = len(self._buckets[bkt_idx])
            self._buckets[bkt_idx].append(sent)
            self._record.append((bkt_idx, idx))

        for bkt_idx, (bucket, length) in enumerate(zip(self._buckets, self._bucket_lengths)):
            self._buckets[bkt_idx] = np.zeros((length, len(bucket), 4), dtype=np.int32)
            for idx, sent in enumerate(bucket):
                self._buckets[bkt_idx][:len(sent), idx, :] = np.array(sent, dtype=np.int32)

    @property
    def idx_sequence(self):
        """
        Indices of sentences when enumerating data set from batches
        :return:
        """
        return [x[1] for x in sorted(zip(self._record, list(range(len(self._record)))))]

    def get_batches(self, batch_size, shuffle=True):
        """
        Get batch iterator
        :param batch_size: size of one batch
        :param shuffle: shuffle batches. Don't set to True when evaluating on dev or test set.
        """
        batches = []
        for bkt_idx, bucket in enumerate(self._buckets):
            bucket_size = bucket.shape[1]
            n_tokens = bucket_size * self._bucket_lengths[bkt_idx]
            n_splits = min(max(n_tokens // batch_size, 1), bucket_size)
            range_func = np.random.permutation if shuffle else np.arange
            for bkt_batch in np.array_split(range_func(bucket_size), n_splits):
                batches.append((bkt_idx, bkt_batch))

        if shuffle:
            np.random.shuffle(batches)

        for bkt_idx, bkt_batch in batches:
            word_inputs = self._buckets[bkt_idx][:, bkt_batch, 0]  # word_id x sent_id
            tag_inputs = self._buckets[bkt_idx][:, bkt_batch, 1]
            arc_targets = self._buckets[bkt_idx][:, bkt_batch, 2]
            rel_targets = self._buckets[bkt_idx][:, bkt_batch, 3]
            yield word_inputs, tag_inputs, arc_targets, rel_targets
