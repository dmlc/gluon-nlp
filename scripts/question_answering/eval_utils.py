"""Modification version of official evaluation script for SQuAD version 2.0.
(https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

"""
import collections
import json
import copy
import re
import string


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

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


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    """
    Compute the token-level f1 scores in which the common tokens are considered
    as True Postives. Precision and recall are percentages of the number of
    common tokens in the prediction and groud truth, respectively.
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred)
                                        for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred)
                                     for a in gold_answers)
    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        # Treat those whose logits exceeds the threshold as unanswerable
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            # The raw scores are converted to 1 if the answerability
            # are predicted else 0
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    """
    Find the best threshold of the raw scores.

    The initial score is set to the number of unanswerable questions,
    assuming that each unanswerable question is successfully predicted.
    In the following traverse, the best threshold is constantly adjusted
    according to the difference from the assumption ('diff').
    """
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    # Rearrange the na_probs in an ascending order, so that the questions
    # with higher probability of answerability the sooner will be read.
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            # For the answerable question
            diff = scores[qid]
        else:
            # For the unanswerable question
            if preds[qid]:
                # Falsely predict the answerability
                diff = -1
            else:
                # Correctly predict the answerability. This is Only true if the
                # prediction is blank, which is no the case before revision
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            # adjust the best thresh over current thresh (na_probs[qid])
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(
        preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(
        preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh


def get_revised_results(preds, na_probs, thresh):
    results = copy.deepcopy(preds)
    for q_id in na_probs.keys():
        if na_probs[q_id] > thresh:
            results[q_id] = ""
    return results


def squad_eval(data_file, preds, na_probs, na_prob_thresh=0.0, revise=False):
    """

    Parameters
    ----------
    data_file
        dataset(list) or data_file(str)
    preds
        predictions dictionary
    na_probs
        probabilities dict of unanswerable
    na_prob_thresh
        threshold of unanswerable
    revise
        Wether to get the final predictions with impossible answers replaced
        with null string ''
    Returns
    -------
        out_eval
            A dictionary of output results
        (preds_out)
            A dictionary of final predictions
    """
    if isinstance(data_file, str):
        with open(data_file) as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']
    elif isinstance(data_file, list):
        dataset = data_file
    if na_probs is None:
        na_probs = {k: 0.0 for k in preds}
        # not necessary to revise results of SQuAD 1.1
        revise = False
    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(dataset, preds)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans,
                                          na_prob_thresh)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans,
                                       na_prob_thresh)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(
            exact_thresh, f1_thresh, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(
            exact_thresh, f1_thresh, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')
        find_all_best_thresh(out_eval, preds, exact_raw,
                             f1_raw, na_probs, qid_to_has_ans)

    if revise:
        thresh = (out_eval['best_exact_thresh'] +
                  out_eval['best_f1_thresh']) * 0.5
        preds_out = get_revised_results(preds, na_probs, thresh)
        return out_eval, preds_out
    else:
        return out_eval, preds
