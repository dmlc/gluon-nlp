"""
In the evaluate function, we use the offical evaluate function as core function.

offical evaluate.py file can be find in the SQuAD dataset offcial web.
"""
from mxnet import autograd, nd
from tqdm import tqdm

try:
    from qanet_config import (CTX, MAX_ANSWER_LENS)
    from official_squad_eval_script import evaluate as official_eval
except ImportError:
    from .qanet_config import (CTX, MAX_ANSWER_LENS)
    from .official_squad_eval_script import evaluate as official_eval


ctx = CTX[0]
ANSWER_MASK_MATRIX = nd.zeros(
    shape=(1, 1000, 1000), ctx=ctx)
for idx in range(MAX_ANSWER_LENS):
    ANSWER_MASK_MATRIX += nd.eye(
        N=1000, M=1000, k=idx, ctx=ctx)


def evaluate(model, dataloader, dataset, original_json_data, ema=None, padding_token_idx=1):
    r"""Evaluate the model on train/dev/test dataset.

    This function is just an encapsulation of official evaluate function.

    The official evaluate code can be find in https://rajpurkar.github.io/SQuAD-explorer/

    Parameters
    ----------
    dataset_type : string, default 'train'
        which dataset to evaluate.
    ema : `ExponentialMovingAverage`
        Whether use the shadow variable to evaluate.
    """
    model.save_parameters('tmp')

    if ema is not None:
        for name, params in model.collect_params().items():
            params.set_data(ema.get_param(name))

    autograd.set_training(False)
    total_answers = {}

    for idxs, context, query, context_char, query_char, _, _ in tqdm(dataloader):
        context = context.as_in_context(ctx)
        query = query.as_in_context(ctx)
        context_char = context_char.as_in_context(ctx)
        query_char = query_char.as_in_context(ctx)

        context_mask = context != padding_token_idx
        query_mask = query != padding_token_idx

        raw_context = [dataset.get_record_by_idx(x.asscalar())[8] for x in idxs]
        spans = [dataset.get_record_by_idx(x.asscalar())[9] for x in idxs]

        begin_hat, end_hat, _, _ = model(
            context, query, context_char, query_char, context_mask, query_mask, None, None)
        begin_hat = begin_hat.softmax(axis=1)
        end_hat = end_hat.softmax(axis=1)

        answer_span_pair = matrix_answer_select(begin_hat, end_hat)
        for i, a, r, s in zip(idxs, answer_span_pair, raw_context, spans):
            total_answers[dataset.get_q_id_by_rec_idx(i.asscalar())] = format_answer(a, r, s)

    model.load_parameters('tmp', ctx=CTX)
    autograd.set_training(True)

    result = official_eval(original_json_data['data'], total_answers)
    f1_score = result['f1']
    em_score = result['exact_match']
    return f1_score, em_score


def matrix_answer_select(begin_hat, end_hat):
    r"""Select the begin and end position of answer span.

        At inference time, the predicted span (s, e) is chosen such that
        begin_hat[s] * end_hat[e] is maximized and s ≤ e.

    Parameters
    ----------
    begin_hat : NDArray
        input tensor with shape `(batch_size, context_sequence_length)`
    end_hat : NDArray
        input tensor with shape `(batch_size, context_sequence_length)`
    """
    global ANSWER_MASK_MATRIX

    begin_hat = begin_hat.reshape(begin_hat.shape + (1,))
    end_hat = end_hat.reshape(end_hat.shape + (1,))
    end_hat = end_hat.transpose(axes=(0, 2, 1))

    result = nd.batch_dot(begin_hat, end_hat) * ANSWER_MASK_MATRIX.slice(
        begin=(0, 0, 0), end=(1, begin_hat.shape[1], begin_hat.shape[1]))

    yp1 = result.max(axis=2).argmax(axis=1, keepdims=True).astype('int32')
    yp2 = result.max(axis=1).argmax(axis=1, keepdims=True).astype('int32')

    return nd.concat(yp1, yp2, dim=-1)


def format_answer(answer_span_pair, context, sp):
    begin = int(answer_span_pair[0].asscalar())
    end = int(answer_span_pair[1].asscalar())

    prediction_text = context[sp[begin][0]:sp[end][1]]
    return prediction_text
