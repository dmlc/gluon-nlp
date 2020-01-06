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

from collections import namedtuple, OrderedDict

from mxnet import nd

_PrelimPrediction = namedtuple(  # pylint: disable=invalid-name
    'PrelimPrediction', [
        'feature_id', 'start_index', 'end_index', 'start_log_prob',
        'end_log_prob'
    ])

_NbestPrediction = namedtuple(  # pylint: disable=invalid-name
    'NbestPrediction', ['text', 'start_log_prob', 'end_log_prob'])


def predict_extended(features,
                     results,
                     n_best_size,
                     max_answer_length=64,
                     start_n_top=5,
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

    prelim_predictions = []
    score_null = 1000000  # large and positive
    for features_id, (result, feature) in enumerate(zip(results, features)):
        cur_null_score = result.cls_logits[0]
        score_null = min(score_null, cur_null_score)
        for i in range(start_n_top):
            for j in range(end_n_top):
                start_log_prob = result.start_top_log_probs[i]
                start_index = int(result.start_top_index[i])
                j_index = j * end_n_top + i
                end_log_prob = result.end_top_log_probs[j_index]
                end_index = int(result.end_top_index[j_index])
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= feature.paragraph_len - 1:
                    continue
                if end_index >= feature.paragraph_len - 1:
                    continue

                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(feature_id=features_id,
                                      start_index=start_index,
                                      end_index=end_index,
                                      start_log_prob=start_log_prob,
                                      end_log_prob=end_log_prob))

    prelim_predictions = sorted(prelim_predictions,
                                key=lambda x:
                                (x.start_log_prob + x.end_log_prob),
                                reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_id]
        tok_start_to_orig_index = feature.tok_start_to_orig_index
        tok_end_to_orig_index = feature.tok_end_to_orig_index
        start_orig_pos = tok_start_to_orig_index[pred.start_index]
        end_orig_pos = tok_end_to_orig_index[pred.end_index]

        paragraph_text = feature.paragraph_text
        final_text = paragraph_text[start_orig_pos:end_orig_pos + 1].strip()
        if final_text in seen_predictions:
            continue
        seen_predictions[final_text] = True
        nbest.append(
            _NbestPrediction(text=final_text,
                             start_log_prob=pred.start_log_prob,
                             end_log_prob=pred.end_log_prob))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(
            _NbestPrediction(text='', start_log_prob=-1e6, end_log_prob=-1e6))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_log_prob + entry.end_log_prob)
        if not best_non_null_entry:
            best_non_null_entry = entry
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
