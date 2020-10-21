import numpy as np
import gluonnlp.data.batchify as bf
import collections
import dataclasses
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from dataclasses import dataclass
from tqdm import tqdm
from typing import Optional, List, Tuple
from gluonnlp.data.tokenizers import BaseTokenizerWithVocab
from gluonnlp.utils.preprocessing import match_tokens_with_char_spans
from gluonnlp.layers import get_activation
from gluonnlp.op import select_vectors_by_position
from gluonnlp.attention_cell import masked_logsoftmax, masked_softmax
import string
import re
import json

int_float_regex = re.compile('^\d+\.{0,1}\d*$')  # matches if a number is either integer or float


RawResultExtended = collections.namedtuple(
    'RawResultExtended',
    ['qas_id',
     'start_top_logits',
     'start_top_index',
     'end_top_logits',
     'end_top_index',
     'answerable_logits'])


def predict_extended(original_feature,
                     chunked_features,
                     results,
                     n_best_size,
                     max_answer_length=64,
                     start_top_n=5,
                     end_top_n=5):
    """Get prediction results for SQuAD.

    Start Logits: (B, N_start)
    End Logits: (B, N_start, N_end)

    Parameters
    ----------
    original_feature:
        The original SquadFeature before chunked
    chunked_features
        List of ChunkFeatures
    results
        List of model predictions for span start and span end.
    n_best_size
        Best N results written to file
    max_answer_length
        Maximum length of the answer tokens.
    start_top_n
        Number of start-position candidates
    end_top_n
        Number of end-position candidates
    Returns
    -------
    not_answerable_score
        Model's estimate that the question is not answerable.
    prediction
        The final prediction.
    nbest_json
        n-best predictions with their probabilities.
    """
    not_answerable_score = 1000000  # Score for not-answerable. We set it to be a large and positive
    # If one chunk votes for answerable, we will treat the context as answerable,
    # Thus, the overall not_answerable_score = min(chunk_not_answerable_score)
    all_start_idx = []
    all_end_idx = []
    all_pred_score = []
    context_length = len(original_feature.context_token_ids)
    token_max_context_score = np.full((len(chunked_features), context_length),
                                      -np.inf,
                                      dtype=np.float32)
    for i, chunked_feature in enumerate(chunked_features):
        chunk_start = chunked_feature.chunk_start
        chunk_length = chunked_feature.chunk_length
        for j in range(chunk_start, chunk_start + chunk_length):
            # This is a heuristic score
            # TODO investigate the impact
            token_max_context_score[i, j] = min(j - chunk_start,
                                                chunk_start + chunk_length - 1 - j) \
                + 0.01 * chunk_length
    token_max_chunk_id = token_max_context_score.argmax(axis=0)

    for chunk_id, (result, chunk_feature) in enumerate(zip(results, chunked_features)):
        # We use the log-likelihood as the not answerable score.
        # Thus, a high score indicates that the answer is not answerable
        cur_not_answerable_score = float(result.answerable_logits[1])
        not_answerable_score = min(not_answerable_score, cur_not_answerable_score)
        # Calculate the start_logits + end_logits as the overall score
        context_offset = chunk_feature.context_offset
        chunk_start = chunk_feature.chunk_start
        chunk_length = chunk_feature.chunk_length
        for i in range(start_top_n):
            for j in range(end_top_n):
                pred_score = result.start_top_logits[i] + result.end_top_logits[i, j]
                start_index = result.start_top_index[i]
                end_index = result.end_top_index[i, j]
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the answer span is in the query tokens or out of
                # the chunk. We throw out all invalid predictions.
                if not (context_offset <= start_index < context_offset + chunk_length) or \
                   not (context_offset <= end_index < context_offset + chunk_length) or \
                   end_index < start_index:
                    continue
                pred_answer_length = end_index - start_index + 1
                if pred_answer_length > max_answer_length:
                    continue
                start_idx = int(start_index - context_offset + chunk_start)
                end_idx = int(end_index - context_offset + chunk_start)
                if token_max_chunk_id[start_idx] != chunk_id:
                    continue
                all_start_idx.append(start_idx)
                all_end_idx.append(end_idx)
                all_pred_score.append(pred_score)
    sorted_start_end_score = sorted(zip(all_start_idx, all_end_idx, all_pred_score),
                                    key=lambda args: args[-1], reverse=True)
    nbest = []
    context_text = original_feature.context_text
    context_token_offsets = original_feature.context_token_offsets
    seen_predictions = set()
    for start_idx, end_idx, pred_score in sorted_start_end_score:
        if len(seen_predictions) >= n_best_size:
            break
        pred_answer = context_text[context_token_offsets[start_idx][0]:
                                   context_token_offsets[end_idx][1]]
        seen_predictions.add(pred_answer)
        nbest.append((pred_answer, pred_score))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if len(nbest) == 0:
        nbest.append(('', float('-inf')))
    all_scores = np.array([ele[1] for ele in nbest], dtype=np.float32)
    probs = np.exp(all_scores) / np.sum(np.exp(all_scores))
    nbest_json = []
    for i, (entry, prob) in enumerate(zip(nbest, probs)):
        output = collections.OrderedDict()
        output['text'] = entry[0]
        output['probability'] = float(prob)
        nbest_json.append(output)

    assert len(nbest_json) >= 1
    return not_answerable_score, nbest[0][0], nbest_json


class ModelForQAConditionalV1(HybridBlock):
    """Here, we use three networks to predict the start scores, end scores and answerable scores.

    We formulate p(start, end, answerable | contextual_embedding) as the product of the
    following three terms:

    - p(start | contextual_embedding)
    - p(end | start, contextual_embedding)
    - p(answerable | contextual_embedding)

    In the inference phase, we are able to use beam search to do the inference.

    use_segmentation is used to mark whether we segment the input sentence. In RoBERTa and XLMR,
    this flag is set to True, then the QA model no longer accept `token_types` as valid input.

    - use_segmentation=True:
        tokens :      <CLS> Question <SEP> Context <SEP>
        token_typess:  0       0       0      1      1

    - use_segmentation=False:
        tokens :      <CLS> Question <SEP> Context <SEP>
        token_typess:  None
    """
    def __init__(self, backbone, units=768, layer_norm_eps=1E-12, dropout_prob=0.1,
                 activation='tanh', weight_initializer=None, bias_initializer=None,
                 use_segmentation=True):
        super().__init__()
        self.backbone = backbone
        self.use_segmentation = use_segmentation
        self.start_scores = nn.Dense(1, flatten=False,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer)
        self.end_scores = nn.HybridSequential()
        self.end_scores.add(nn.Dense(units, flatten=False,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer))
        self.end_scores.add(get_activation(activation))
        self.end_scores.add(nn.LayerNorm(epsilon=layer_norm_eps))
        self.end_scores.add(nn.Dense(1, flatten=False,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer))
        self.answerable_scores = nn.HybridSequential()
        self.answerable_scores.add(nn.Dense(units, flatten=False,
                                            weight_initializer=weight_initializer,
                                            bias_initializer=bias_initializer))
        self.answerable_scores.add(get_activation(activation))
        self.answerable_scores.add(nn.Dropout(dropout_prob))
        self.answerable_scores.add(nn.Dense(2, flatten=False,
                                            weight_initializer=weight_initializer,
                                            bias_initializer=bias_initializer))

    def get_start_logits(self, F, contextual_embedding, p_mask):
        """

        Parameters
        ----------
        F
        contextual_embedding
            Shape (batch_size, sequence_length, C)

        Returns
        -------
        start_logits
            Shape (batch_size, sequence_length)
        """
        start_scores = F.np.squeeze(self.start_scores(contextual_embedding), -1)
        start_logits = masked_logsoftmax(F, start_scores, mask=p_mask, axis=-1)
        return start_logits

    def get_end_logits(self, F, contextual_embedding, start_positions, p_mask):
        """

        Parameters
        ----------
        F
        contextual_embedding
            Shape (batch_size, sequence_length, C)
        start_positions
            Shape (batch_size, N)
            We process multiple candidates simultaneously
        p_mask
            Shape (batch_size, sequence_length)

        Returns
        -------
        end_logits
            Shape (batch_size, N, sequence_length)
        """
        # Select the features at the start_positions
        # start_feature will have shape (batch_size, N, C)
        start_features = select_vectors_by_position(F, contextual_embedding, start_positions)
        # Concatenate the start_feature and the contextual_embedding
        contextual_embedding = F.np.expand_dims(contextual_embedding, axis=1)  # (B, 1, T, C)
        start_features = F.np.expand_dims(start_features, axis=2)  # (B, N, 1, C)
        concat_features = F.np.concatenate([F.npx.broadcast_like(start_features,
                                                                 contextual_embedding, 2, 2),
                                            F.npx.broadcast_like(contextual_embedding,
                                                                 start_features, 1, 1)],
                                           axis=-1)  # (B, N, T, 2C)
        end_scores = self.end_scores(concat_features)
        end_scores = F.np.squeeze(end_scores, -1)
        end_logits = masked_logsoftmax(F, end_scores, mask=F.np.expand_dims(p_mask, axis=1),
                                       axis=-1)
        return end_logits

    def get_answerable_logits(self, F, contextual_embedding, p_mask):
        """Get the answerable logits.

        Parameters
        ----------
        F
        contextual_embedding
            Shape (batch_size, sequence_length, C)
        p_mask
            Shape (batch_size, sequence_length)
            Mask the sequence.
            0 --> Denote that the element is masked,
            1 --> Denote that the element is not masked

        Returns
        -------
        answerable_logits
            Shape (batch_size, 2)
        """
        # Shape (batch_size, sequence_length)
        start_scores = F.np.squeeze(self.start_scores(contextual_embedding), -1)
        start_score_weights = masked_softmax(F, start_scores, p_mask, axis=-1)
        start_agg_feature = F.npx.batch_dot(F.np.expand_dims(start_score_weights, axis=1),
                                            contextual_embedding)
        start_agg_feature = F.np.squeeze(start_agg_feature, 1)
        cls_feature = contextual_embedding[:, 0, :]
        answerable_scores = self.answerable_scores(F.np.concatenate([start_agg_feature,
                                                                     cls_feature], axis=-1))
        answerable_logits = F.npx.log_softmax(answerable_scores, axis=-1)
        return answerable_logits

    def hybrid_forward(self, F, tokens, token_types, valid_length, p_mask, start_position):
        """

        Parameters
        ----------
        F
        tokens
            Shape (batch_size, sequence_length)
        token_types
            Shape (batch_size, sequence_length)
        valid_length
            Shape (batch_size,)
        p_mask
            Shape (batch_size, sequence_length)
        start_position
            Shape (batch_size,)

        Returns
        -------
        start_logits
            Shape (batch_size, sequence_length)
        end_logits
            Shape (batch_size, sequence_length)
        answerable_logits
        """
        if self.use_segmentation:
            contextual_embeddings = self.backbone(tokens, token_types, valid_length)
        else:
            contextual_embeddings = self.backbone(tokens, valid_length)
        start_logits = self.get_start_logits(F, contextual_embeddings, p_mask)
        end_logits = self.get_end_logits(F, contextual_embeddings,
                                         F.np.expand_dims(start_position, axis=1),
                                         p_mask)
        end_logits = F.np.squeeze(end_logits, axis=1)
        answerable_logits = self.get_answerable_logits(F, contextual_embeddings, p_mask)
        return start_logits, end_logits, answerable_logits

    def inference(self, tokens, token_types, valid_length, p_mask,
                  start_top_n: int = 5, end_top_n: int = 5):
        """Get the inference result with beam search

        Parameters
        ----------
        tokens
            The input tokens. Shape (batch_size, sequence_length)
        token_types
            The input token types. Shape (batch_size, sequence_length)
        valid_length
            The valid length of the tokens. Shape (batch_size,)
        p_mask
            The mask which indicates that some tokens won't be used in the calculation.
            Shape (batch_size, sequence_length)
        start_top_n
            The number of candidates to select for the start position.
        end_top_n
            The number of candidates to select for the end position.

        Returns
        -------
        start_top_logits
            The top start logits
            Shape (batch_size, start_top_n)
        start_top_index
            Index of the top start logits
            Shape (batch_size, start_top_n)
        end_top_logits
            The top end logits.
            Shape (batch_size, start_top_n, end_top_n)
        end_top_index
            Index of the top end logits
            Shape (batch_size, start_top_n, end_top_n)
        answerable_logits
            The answerable logits. Here 0 --> answerable and 1 --> not answerable.
            Shape (batch_size, sequence_length, 2)
        """
        # Shape (batch_size, sequence_length, C)
        if self.use_segmentation:
            contextual_embeddings = self.backbone(tokens, token_types, valid_length)
        else:
            contextual_embeddings = self.backbone(tokens, valid_length)
        start_logits = self.get_start_logits(mx.nd, contextual_embeddings, p_mask)
        # The shape of start_top_index will be (..., start_top_n)
        start_top_logits, start_top_index = mx.npx.topk(start_logits, k=start_top_n, axis=-1,
                                                        ret_typ='both')
        end_logits = self.get_end_logits(mx.nd, contextual_embeddings, start_top_index, p_mask)
        # Note that end_top_index and end_top_log_probs have shape (bsz, start_n_top, end_n_top)
        # So that for each start position, there are end_n_top end positions on the third dim.
        end_top_logits, end_top_index = mx.npx.topk(end_logits, k=end_top_n, axis=-1,
                                                    ret_typ='both')
        answerable_logits = self.get_answerable_logits(mx.nd, contextual_embeddings, p_mask)
        return start_top_logits, start_top_index, end_top_logits, end_top_index, \
                    answerable_logits


@dataclass
class DocChunk:
    start: int
    length: int
    is_impossible: bool
    gt_start_pos: Optional[int]
    gt_end_pos: Optional[int]


@dataclass
class SquadFeature:
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
    qas_id: int
    query_token_ids: List[int]
    context_text: str
    context_token_ids: List[int]
    context_token_offsets: List[Tuple[int, int]]
    is_impossible: bool
    token_answer_mismatch: bool
    unreliable_span: bool
    gt_answer_text: str
    gt_start_pos: Optional[int]
    gt_end_pos: Optional[int]

    def to_json(self):
        return json.dumps(dataclasses.asdict(self),
                          ensure_ascii=False)

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



@dataclass
class SquadExample:
    """A single training/test example for the Squad dataset, as loaded from disk.

    Attributes
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
    qas_id: int
    query_text: str
    context_text: str
    answer_text: str
    start_position: int
    end_position: int
    title: str
    answers: Optional[List[str]] = None
    is_impossible: bool = False

    def to_json(self):
        return json.dumps(dataclasses.asdict(self),
                          ensure_ascii=False)

    @classmethod
    def from_json(cls, s):
        kwargs = json.loads(s)
        return cls(**kwargs)


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
    np_offsets = np.array(offsets)
    if is_training and not example.is_impossible:
        assert example.start_position >= 0 and example.end_position >= 0
        # We convert the character-level offsets to token-level offsets
        # Also, if the answer after tokenization + detokenization is not the same as the original
        # answer, we try to localize the answer text and do a rematch
        candidates = [(example.start_position, example.end_position)]
        all_possible_start_pos = {example.start_position}
        find_all_candidates = False
        lower_idx, upper_idx = None, None
        first_lower_idx, first_upper_idx = None, None
        while len(candidates) > 0:
            start_position, end_position = candidates.pop()
            # Match the token offsets
            token_start_ends = match_tokens_with_char_spans(np_offsets,
                                                            np.array([[start_position,
                                                                       end_position]]))
            lower_idx = int(token_start_ends[0][0])
            upper_idx = int(token_start_ends[0][1])
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



class SquadDatasetProcessor:

    def __init__(self, tokenizer, doc_stride, max_seq_length, max_query_length):
        """

        Parameters
        ----------
        tokenizer
            The tokenizer
        doc_stride
            The stride to chunk the document
        max_seq_length
            Maximum length of the merged data
        max_query_length
            Maximum query length
        """
        self._tokenizer = tokenizer
        self._doc_stride = doc_stride
        self._max_seq_length = max_seq_length
        self._max_query_length = max_query_length

        vocab = tokenizer.vocab
        self.pad_id = vocab.pad_id
        # For roberta model, taking sepecial token <s> as [CLS] and </s> as [SEP]
        self.cls_id = vocab.bos_id if 'cls_token' not in vocab.special_token_keys else vocab.cls_id
        self.sep_id = vocab.eos_id if 'sep_token' not in vocab.special_token_keys else vocab.sep_id

        # TODO(sxjscience) Consider to combine the NamedTuple and batchify functionality.
        self.ChunkFeature = collections.namedtuple('ChunkFeature',
                                              ['qas_id',
                                               'data',
                                               'valid_length',
                                               'segment_ids',
                                               'masks',
                                               'is_impossible',
                                               'gt_start',
                                               'gt_end',
                                               'context_offset',
                                               'chunk_start',
                                               'chunk_length'])
        self.BatchifyFunction = bf.NamedTuple(self.ChunkFeature,
                                         {'qas_id': bf.List(),
                                          'data': bf.Pad(val=self.pad_id),
                                          'valid_length': bf.Stack(),
                                          'segment_ids': bf.Pad(),
                                          'masks': bf.Pad(val=1),
                                          'is_impossible': bf.Stack(),
                                          'gt_start': bf.Stack(),
                                          'gt_end': bf.Stack(),
                                          'context_offset': bf.Stack(),
                                          'chunk_start': bf.Stack(),
                                          'chunk_length': bf.Stack()})

    def process_sample(self, feature: SquadFeature):
        """Process the data to the following format.

        Note that we mask all the special tokens except the CLS token. The reason for not masking
        the CLS token is that if the question is not answerable, we will set the start and end to
        be 0.


        Merged:      <CLS> Question <SEP> Context <SEP>
        Segment IDs:  0       0       0      1      1
        Mask:         0       1       1      0      1

        Here, we need to emphasize that when mask = 1, the data are actually not masked!

        Parameters
        ----------
        feature
            Tokenized SQuAD feature

        Returns
        -------
        ret
            Divide the feature into multiple chunks and extract the feature which contains
            the following:
            - data
                The data that concatenates the query and the context + special tokens
            - valid_length
                The valid_length of the data
            - segment_ids
                We assign the query part as segment 0 and the context part as segment 1.
            - masks
                We mask all the special tokens. 1 --> not masked, 0 --> masked.
            - is_impossible
                Whether the provided context is impossible to answer or not.
            - gt_start
                The ground-truth start location of the span
            - gt_end
                The ground-truth end location of the span
            - chunk_start
                The start of the chunk
            - chunk_length
                The length of the chunk
        """
        ret = []
        truncated_query_ids = feature.query_token_ids[:self._max_query_length]
        chunks = feature.get_chunks(
            doc_stride=self._doc_stride,
            max_chunk_length=self._max_seq_length - len(truncated_query_ids) - 3)
        for chunk in chunks:
            data = np.array([self.cls_id] + truncated_query_ids + [self.sep_id] +
                            feature.context_token_ids[chunk.start:(chunk.start + chunk.length)] +
                            [self.sep_id], dtype=np.int32)
            valid_length = len(data)
            segment_ids = np.array([0] + [0] * len(truncated_query_ids) +
                                   [0] + [1] * chunk.length + [1], dtype=np.int32)
            masks = np.array([0] + [1] * len(truncated_query_ids) + [1] + [0] * chunk.length + [1],
                             dtype=np.int32)
            context_offset = len(truncated_query_ids) + 2
            if chunk.gt_start_pos is None and chunk.gt_end_pos is None:
                start_pos = 0
                end_pos = 0
            else:
                # Here, we increase the start and end because we put query before context
                start_pos = chunk.gt_start_pos + context_offset
                end_pos = chunk.gt_end_pos + context_offset
            chunk_feature = self.ChunkFeature(qas_id=feature.qas_id,
                                              data=data,
                                              valid_length=valid_length,
                                              segment_ids=segment_ids,
                                              masks=masks,
                                              is_impossible=chunk.is_impossible,
                                              gt_start=start_pos,
                                              gt_end=end_pos,
                                              context_offset=context_offset,
                                              chunk_start=chunk.start,
                                              chunk_length=chunk.length)
            ret.append(chunk_feature)
        return ret

    def get_train(self, features, skip_unreliable=True):
        """Get the training dataset

        Parameters
        ----------
        features
        skip_unreliable
            Whether to skip the unreliable spans in the training set

        Returns
        -------
        train_dataset
        num_token_answer_mismatch
        num_unreliable
        """
        train_dataset = []
        num_token_answer_mismatch = 0
        num_unreliable = 0
        for feature in features:
            if feature.token_answer_mismatch:
                num_token_answer_mismatch += 1
            if feature.unreliable_span:
                num_unreliable += 1
            if skip_unreliable and feature.unreliable_span:
                # Skip when not reliable
                continue
            # Process the feature
            chunk_features = self.process_sample(feature)
            train_dataset.extend(chunk_features)
        return train_dataset, num_token_answer_mismatch, num_unreliable
    
