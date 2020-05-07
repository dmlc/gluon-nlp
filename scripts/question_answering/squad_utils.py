"""Utility classes and functions for data processing"""
from typing import Optional, List
from collections import namedtuple
import itertools
import bisect
import re
import numpy as np
import numpy.ma as ma
import warnings
import os
from tqdm import tqdm
import json
import string
from gluonnlp.data.tokenizers import BaseTokenizerWithVocab
from typing import Tuple
from mxnet.gluon.utils import download

int_float_regex = re.compile('^\d+\.{0,1}\d*$')  # matches if a number is either integer or float

import mxnet as mx
mx.npx.set_np()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.
    This is from the official evaluate-v2.0.py in SQuAD.
    """

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_official_squad_eval_script(version='2.0', download_dir=None):
    url_info = {'2.0': ['evaluate-v2.0.py',
                        'https://worksheets.codalab.org/rest/bundles/'
                        '0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/',
                        '5a584f1952c88b4088be5b51f2046a2c337aa706']}
    if version not in url_info:
        raise ValueError('Version {} is not supported'.format(version))
    if download_dir is None:
        download_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
    download_path = os.path.join(download_dir, url_info[version][0])
    download(url_info[version][1], download_path, sha1_hash=url_info[version][2])
    return download_path


class SquadExample:
    """A single training/test example for the Squad dataset, as loaded from disk."""
    def __init__(self, qas_id: int,
                 query_text: str,
                 context_text: str,
                 answer_text: str,
                 start_position: int,
                 end_position: int,
                 title: str,
                 answers: Optional[List[str]] = None,
                 is_impossible: bool = False):
        """

        Parameters
        ----------
        qas_id
            The example's unique identifier
        query_text
            The query string
        context_text
            The context string
        answer_text
            The answer string
        start_position
            The character position of the start of the answer
        end_position
            The character position of the end of the answer
        title
            The title of the example
        answers
            None by default, this is used during evaluation.
            Holds answers as well as their start positions.
        is_impossible
            False by default, set to True if the example has no possible answer.
        """
        self.qas_id = qas_id
        self.query_text = query_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers
        self.start_position = start_position
        self.end_position = end_position

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, s):
        kwargs = json.loads(s)
        return cls(**kwargs)


DocChunk = namedtuple('DocChunk', ['start', 'length',
                                   'is_impossible',
                                   'gt_start_pos',
                                   'gt_end_pos'])


class SquadFeature:
    def __init__(self, qas_id,
                 query_token_ids,
                 context_text,
                 context_token_ids,
                 context_token_offsets,
                 is_impossible,
                 token_answer_mismatch,
                 unreliable_span,
                 gt_answer_text,
                 gt_start_pos,
                 gt_end_pos):
        """The Squad Feature

        Parameters
        ----------
        qas_id
            The unique query/answer ID in the squad dataset
        query_token_ids
            The tokenized query.
        context_text
            The original text of the context
        context_token_ids
            The tokenized context.
        context_token_offsets
            The offsets of the tokens in the original context string
        is_impossible
            Whether the sample is impossible.
        token_answer_mismatch
            If this value is True, it means that we cannot reconstruct the ground-truth answer with
            the tokenized version. Usually, the span-prediction-based approach won't be very
            accurate and we should rely on the encoder-decoder approach.
            For example:
                GT: "japan", Tokenized Version: "japanese"
                     "six'                       "sixth"
                     "one"                       "iPhone"
                     "breed"                     "breeding"
                     "emotion"                   "emotional"

        unreliable_span
            If this value is True, it means that we cannot rely on the gt_start_pos and gt_end_pos.
            In this scenario, we cannot utilize the span-prediction-based approach.
            One example is the question about "how many", the answer will spread across the
            whole document and there is no clear span.
        gt_answer_text
            The ground-truth answer text
        gt_start_pos
            The start position of the ground-truth span. None indicates that there is no valid
            ground-truth span.
        gt_end_pos
            The end position of the ground-truth span. None indicates that there is no valid
            ground-truth span.
        """
        self.qas_id = qas_id
        self.query_token_ids = query_token_ids
        self.context_text = context_text
        self.context_token_ids = context_token_ids
        self.context_token_offsets = context_token_offsets
        self.is_impossible = is_impossible
        self.token_answer_mismatch = token_answer_mismatch
        self.unreliable_span = unreliable_span
        self.gt_answer_text = gt_answer_text
        self.gt_start_pos = gt_start_pos
        self.gt_end_pos = gt_end_pos

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, s):
        kwargs = json.loads(s)
        return cls(**kwargs)

    def __repr__(self):
        return self.to_json()

    def get_chunks(self, doc_stride, max_chunk_length=None):
        """Get a sequence of chunks for the squad feature.

        In reality, the document will be too long for the NLP model, and we will split it into
        multiple chunks.

        For example, consider the following
        Doc: the man went to the store and bought a gallon of milk

        We may divide it into four chunks:

        Chunk 1: the man went to the
        Chunk 2: to the store and bought
        Chunk 3: and bought a gallon of
        Chunk 4: gallon of milk

        We will use our network to extract features for each chunk,
        and do the aggregation afterwards. Here, one token may appear in multiple chunks.
        We can vote the output based on some heuristic score functions.

        Parameters
        ----------
        doc_stride
            The stride used when the context is too large and is split across several features.
        max_chunk_length
            The maximum size of the chunk

        Returns
        -------
        ret
            List of DocChunk objects
        """
        doc_ptr = 0
        max_chunk_length = max_chunk_length if max_chunk_length is not None else \
            len(self.context_token_ids)
        ret = []
        while doc_ptr < len(self.context_token_ids):
            chunk_length = min(max_chunk_length, len(self.context_token_ids) - doc_ptr)
            if self.gt_answer_text is None:
                chunk_gt_start_pos = None
                chunk_gt_end_pos = None
                chunk_is_impossible = True
            else:
                if self.gt_start_pos is not None and self.gt_end_pos is not None and\
                        self.gt_start_pos >= doc_ptr and self.gt_end_pos < doc_ptr + chunk_length:
                    # The chunk contains the ground-truth annotation
                    chunk_gt_start_pos = self.gt_start_pos - doc_ptr
                    chunk_gt_end_pos = self.gt_end_pos - doc_ptr
                    chunk_is_impossible = False
                else:
                    chunk_gt_start_pos = None
                    chunk_gt_end_pos = None
                    chunk_is_impossible = True
            ret.append(DocChunk(start=doc_ptr,
                                length=chunk_length,
                                is_impossible=chunk_is_impossible,
                                gt_start_pos=chunk_gt_start_pos,
                                gt_end_pos=chunk_gt_end_pos))
            if doc_ptr + chunk_length == len(self.context_token_ids):
                break
            doc_ptr += doc_stride
        return ret


def get_squad_examples_from_json(json_file: str, is_training: bool) -> List[SquadExample]:
    """
    Read the whole entry of raw json file and convert it to examples.

    Parameters
    ----------
    json_file
        The path to the json file
    is_training
        Whether or not training

    Returns
    -------
    ret
        List of SquadExample objects
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    examples = []
    for entry in tqdm(data['data']):
        title = entry['title']
        for paragraph in entry['paragraphs']:
            context_text = paragraph['context']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                query_text = qa['question']
                start_position = None
                end_position = None
                answer_text = None
                answers = None
                if "is_impossible" in qa:
                    is_impossible = qa["is_impossible"]
                else:
                    is_impossible = False

                if not is_impossible:
                    if is_training:
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position = answer["answer_start"]
                        end_position = start_position + len(answer_text)
                        if context_text[start_position:end_position] != answer_text:
                            warnings.warn(
                                'Mismatch start/end and answer_text, start/end={}/{},'
                                ' answer text={}. qas={}'
                                .format(start_position, end_position, answer_text, qas_id))
                    else:
                        answers = qa["answers"]
                example = SquadExample(
                    qas_id=qas_id,
                    query_text=query_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )
                examples.append(example)
    return examples


def get_squad_examples(data_dir, segment='train', version='1.1'):
    """

    Parameters
    ----------
    data_dir
        The directory of the data
    segment
        The segment
    version
        Version of the SQuAD

    Returns
    -------
    examples
        A list of SquadExampls objects
    """
    if version == '1.1':
        train_path = os.path.join(data_dir, 'train-v1.1.json')
        dev_path = os.path.join(data_dir, 'dev-v1.1.json')
    elif version == '2.0':
        train_path = os.path.join(data_dir, 'train-v2.0.json')
        dev_path = os.path.join(data_dir, 'dev-v2.0.json')
    else:
        raise NotImplementedError

    if segment == 'train':
        examples = get_squad_examples_from_json(train_path, is_training=True)
    elif segment == 'dev':
        examples = get_squad_examples_from_json(dev_path, is_training=False)
    else:
        raise NotImplementedError

    return examples


def convert_squad_example_to_feature(example: SquadExample,
                                     tokenizer: BaseTokenizerWithVocab,
                                     is_training: bool):
    """
    Convert a SquadExample object to a SquadFeature object with the designated tokenizer.

    There are accually few examples can not be converted properly with token level tokenization,
    due to the ground-truth are given by the start position and the answer text, and some examples
    are annotated with wrong labels. Thus, attribute unreliable_span and token_answer_mismatch are
    used to indicate these senarios.

    Parameters
    ----------
    example
        A single squad example
    tokenizer
        The trained tokenizer
    is_training
        Whether to deal with the training case
    Returns
    -------
    feature
        A SquadFeature
    """
    context_text = example.context_text
    answer_text = example.answer_text
    query_text = example.query_text
    context_token_ids, offsets = tokenizer.encode_with_offsets(context_text, int)
    query_token_ids = tokenizer.encode(query_text, int)
    gt_answer_text = answer_text
    gt_span_start_pos, gt_span_end_pos = None, None
    token_answer_mismatch = False
    unreliable_span = False
    if is_training and not example.is_impossible:
        assert example.start_position >= 0 and example.end_position >= 0
        # From the offsets, we locate the first offset that contains start_pos and the last offset
        # that contains end_pos, i.e.
        # offsets[lower_idx][0] <= start_pos < offsets[lower_idx][1]
        # offsets[upper_idx][0] < end_pos <= offsets[upper_idx[1]
        # Also, if the answer after tokenization + detokenization is not the same as the original
        # answer,
        offsets_lower = [offset[0] for offset in offsets]
        offsets_upper = [offset[1] for offset in offsets]
        candidates = [(example.start_position, example.end_position)]
        all_possible_start_pos = {example.start_position}
        find_all_candidates = False
        lower_idx, upper_idx = None, None
        first_lower_idx, first_upper_idx = None, None
        while len(candidates) > 0:
            start_position, end_position = candidates.pop()
            if end_position > offsets_upper[-1] or start_position < offsets_lower[0]:
                # Detect the out-of-boundary case
                warnings.warn('The selected answer is not covered by the tokens! '
                              'Use the end_position. '
                              'qas_id={}, context_text={}, start_pos={}, end_pos={}, '
                              'offsets={}'.format(example.qas_id, context_text,
                                                  start_position, end_position, offsets))
                end_position = min(offsets_upper[-1], end_position)
                start_position = max(offsets_upper[0], start_position)
            lower_idx = bisect.bisect(offsets_lower, start_position) - 1
            upper_idx = bisect.bisect_left(offsets_upper, end_position)
            if not find_all_candidates:
                first_lower_idx = lower_idx
                first_upper_idx = upper_idx
            # The new start pos and end_pos are the lower_idx and upper_idx
            sliced_answer = context_text[offsets[lower_idx][0]:offsets[upper_idx][1]]
            norm_sliced_answer = normalize_answer(sliced_answer)
            norm_answer = normalize_answer(answer_text)
            if norm_sliced_answer != norm_answer:
                if not find_all_candidates:
                    # Try to find a better start+end of the answer and insert all positions to the
                    # candidates
                    find_all_candidates = True
                    pos = context_text.find(answer_text)
                    while pos != -1:
                        if pos not in all_possible_start_pos:
                            all_possible_start_pos.add(pos)
                            candidates.append((pos, pos + len(answer_text)))
                        pos = context_text.find(answer_text, pos + 1)
                elif len(candidates) == 0:
                    token_answer_mismatch = True
                    lower_idx = first_lower_idx
                    upper_idx = first_upper_idx
                    if int_float_regex.match(answer_text):
                        # Find an integer/float and the sample won't be reliable.
                        # The span-based approach is not suitable for this scenario and we will
                        # set the unreliable span flag.
                        unreliable_span = True
            else:
                break

        gt_span_start_pos = lower_idx
        gt_span_end_pos = upper_idx

    feature = SquadFeature(qas_id=example.qas_id,
                           query_token_ids=query_token_ids,
                           context_text=context_text,
                           context_token_ids=context_token_ids,
                           context_token_offsets=offsets,
                           is_impossible=example.is_impossible,
                           token_answer_mismatch=token_answer_mismatch,
                           unreliable_span=unreliable_span,
                           gt_answer_text=gt_answer_text,
                           gt_start_pos=gt_span_start_pos,
                           gt_end_pos=gt_span_end_pos)
    return feature
