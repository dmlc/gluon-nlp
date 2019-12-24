# Copyright 2018 The Google AI Language Team Authors, Allenai and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XLNet SQuAD evaluate."""

import json
import math
import collections
from collections import namedtuple, OrderedDict

from mxnet import nd
from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores
from model.temp_utils import BasicTokenizer

def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def predict_extended(features, results, tokenizer, n_best_size, max_answer_length=64, start_n_top=5,
                     end_n_top=5):
    """Get prediction results for XLNet.

    Parameters
    ----------
    features : list of SQuADFeature
        List of squad features for the example.
    results : list of data.qa.PredResult
        List of model predictions for span start and span end.
    tokenizer: callable
        Tokenizer function.
    max_answer_length: int, default 64
        Maximum length of the answer tokens.
    null_score_diff_threshold: float, default 0.0
        If null_score - best_non_null is greater than the threshold predict null.
    n_best_size: int, default 10
        The total number of n-best predictions.
    version_2: bool, default False
        If true, the SQuAD examples contain some that do not have an answer.

    Returns
    -------
    prediction: str
        The final prediction.
    nbest : list of (str, float)
        n-best predictions with their probabilities.
    """

    _PrelimPrediction = namedtuple(  # pylint: disable=invalid-name
        'PrelimPrediction',
        ['features_id', 'start_index', 'end_index', 'start_log_prob', 'end_log_prob'])

    _NbestPrediction = namedtuple(  # pylint: disable=invalid-name
        'NbestPrediction', ['text', 'start_log_prob', 'end_log_prob'])

    prelim_predictions = []
    score_null = 1000000  # large and positive

    for features_id, (result, feature) in enumerate(zip(results, features)):
        cur_null_score = result.cls_logits[0]
        score_null = min(score_null, cur_null_score)
        #print("id: ", feature.qas_id)
        #print("len: ", len(feature.tokens))
        #print("feature answer :", feature.orig_answer_text)
        #print("featture isimpossible: ", feature.is_impossible)
        #print("start index: ", result.start_top_index)
        #for i in range(start_n_top):
        #    print("maybe: ", feature.tokens[int(result.start_top_index[i])])
        #print("end index: ", result.end_top_index)
        #print("start log: ", result.start_top_log_probs)
        #print("end log: " ,result.end_top_log_probs)
        for i in range(start_n_top):
            for j in range(end_n_top):
                start_log_prob = result.start_top_log_probs[i]
                start_index = int(result.start_top_index[i])
                j_index = i * end_n_top + j
                end_log_prob = result.end_top_log_probs[j_index]
                end_index = int(result.end_top_index[j_index])
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map.keys():
                    continue
                if end_index not in feature.token_to_orig_map.keys():
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                #print("st: {0}, ed: {1}".format(start_index, end_index))
                prelim_predictions.append(
                    _PrelimPrediction(features_id=features_id, start_index=start_index,
                                      end_index=end_index, start_log_prob=start_log_prob,
                                      end_log_prob=end_log_prob))

    prelim_predictions = sorted(prelim_predictions, key=lambda x:
                                (x.start_log_prob + x.end_log_prob), reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.features_id]
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = feature.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = ' '.join(tok_tokens)
            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(' ##', '')
            tok_text = tok_text.replace('##', '')

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = ' '.join(tok_text.split())
            orig_text = ' '.join(orig_tokens)
            #print("tok_text: ", tok_text)
            #print("orig text: ", orig_text)
            final_text = get_final_text(tok_text, orig_text, tokenizer)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ''
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(text=final_text, start_log_prob=pred.start_log_prob,
                             end_log_prob=pred.end_log_prob))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(_NbestPrediction(text='', start_log_prob=-1e6, end_log_prob=-1e6))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_log_prob + entry.end_log_prob)
        if not best_non_null_entry:
            best_non_null_entry = entry
    print("best: ", best_non_null_entry)
    print("--------------------------")
    probs = nd.softmax(nd.array(total_scores)).asnumpy()

    nbest_json = []

    for (i, entry) in enumerate(nbest):
        output = OrderedDict()
        output['text'] = entry.text
        output['probability'] = float(probs[i])
        output['start_log_prob'] = float(entry.start_log_prob)
        output['end_log_prob'] = float(entry.end_log_prob)
        nbest_json.append(output)

    assert len(nbest_json) >= 1
    assert best_non_null_entry is not None
    score_diff = score_null
    return score_diff, best_non_null_entry.text, nbest_json


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
