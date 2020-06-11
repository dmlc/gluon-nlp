"""
Question Answering Ensemble with Pretrained Language Model
"""
# pylint:disable=redefined-outer-name,logging-format-interpolation

import os
import json
import time
import logging
import argparse
import functools
import collections
from multiprocessing import Pool, cpu_count

import mxnet as mx
import numpy as np

from models import ModelForQAConditionalV1
from run_squad import RawResultExtended, SquadDatasetProcessor
from eval_utils import squad_eval
from squad_utils import SquadFeature, get_squad_examples, convert_squad_example_to_feature, ml_voter
from gluonnlp.utils.misc import grouper, set_seed, parse_ctx, logging_config
from gluonnlp.initializer import TruncNorm

mx.npx.set_np()

CACHE_PATH = os.path.realpath(os.path.join(os.path.realpath(__file__), '..', 'cached'))
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Question Answering example.'
                    ' We fine-tune the pretrained model on SQuAD dataset.')
    parser.add_argument('--model_name', type=str, default='google_albert_base_v2',
                        help='Name of the pretrained model.')
    parser.add_argument('--do_ensemble', action='store_true',
                        help='Whether to ensemble the model')
    parser.add_argument('--data_dir', type=str, default='squad')
    parser.add_argument('--version', default='2.0', choices=['1.1', '2.0'],
                        help='Version of the SQuAD dataset.')
    parser.add_argument('--output_dir', type=str, default='squad_out',
                        help='The output directory where the model params will be written.'
                             ' default is squad_out')
    parser.add_argument('--voter_path', type=str, default=None,
                        help='Path to the parameter of voter')
    parser.add_argument('--gpus', type=str, default='0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. -1 means using cpu.')
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Evaluate batch size. Number of examples per gpu in a minibatch for '
                             'evaluation.')
    parser.add_argument('--classifier_dropout', type=float, default=0.1,
                        help='dropout of classifier')
    # Data pre/post processing
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='The maximum total input sequence length after tokenization.'
                             'Sequences longer than this will be truncated, and sequences shorter '
                             'than this will be padded. default is 512')
    parser.add_argument('--doc_stride', type=int, default=128,
                        help='When splitting up a long document into chunks, how much stride to '
                             'take between chunks. default is 128')
    parser.add_argument('--max_query_length', type=int, default=64,
                        help='The maximum number of tokens for the query. Questions longer than '
                             'this will be truncated to this length. default is 64')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Whether to overwrite the feature cache.')
    # Evaluation hyperparameters
    parser.add_argument('--start_top_n', type=int, default=5,
                        help='Number of start-position candidates')
    parser.add_argument('--end_top_n', type=int, default=5,
                        help='Number of end-position candidates corresponding '
                             'to a start position')
    parser.add_argument('--n_best_size', type=int, default=20, help='Top N results written to file')
    parser.add_argument(
        '--max_answer_length',
        type=int,
        default=30,
        help='The maximum length of an answer that can be generated. This is needed '
             'because the start and end predictions are not conditioned on one another.'
             ' default is 30')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='The path of checkpoints to be ensembled')
    args = parser.parse_args()
    return args


def get_network(model_name, dropout=0.1):
    if 'albert' in model_name:
        from gluonnlp.models.albert import AlbertModel, get_pretrained_albert
        Model, get_pretrained_model = AlbertModel, get_pretrained_albert
    elif 'bert' in model_name:
        from gluonnlp.models.bert import BertModel, get_pretrained_bert
        Model, get_pretrained_model = BertModel, get_pretrained_bert
    elif 'electra' in model_name:
        from gluonnlp.models.electra import ElectraModel, get_pretrained_electra
        Model, get_pretrained_model = ElectraModel, get_pretrained_electra
    else:
        raise NotImplementedError()
    # Create the network
    cfg, tokenizer, _, _ = get_pretrained_model(model_name, load_backbone=False)
    cfg = Model.get_cfg().clone_merge(cfg)
    backbone = Model.from_cfg(cfg, use_pooler=False)

    qa_net = ModelForQAConditionalV1(backbone=backbone,
                                     dropout_prob=dropout,
                                     weight_initializer=TruncNorm(stdev=0.02),
                                     prefix='qa_net_')
    qa_net.hybridize()

    return cfg, tokenizer, qa_net


SinglePredict = collections.namedtuple(
    'SinglePredict',
    ['start_idx',
     'end_idx',
     'has_score',
     'no_score',
     'cls_score'])


def predict_extended(feature,
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
    feature:
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
    # If one chunk votes for answerable, we will treat the context as answerable,
    # Thus, the overall not_answerable_score = min(chunk_not_answerable_score)
    start_end = []
    context_length = len(feature.context_token_ids)
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
        # Thus, a high score indicates that the answer is not answerable.
        # The second item is an additional logits represents the sum of
        # logits of the cls token in start and end positions.
        cur_not_answerable_score = float(result.answerable_logits[1])
        pos_cls_score = float(result.pos_cls_logits)
        # Calculate the start_logits + end_logits as the overall score
        context_offset = chunk_feature.context_offset
        chunk_start = chunk_feature.chunk_start
        chunk_length = chunk_feature.chunk_length
        for i in range(start_top_n):
            for j in range(end_top_n):
                has_score = result.start_top_logits[i] + result.end_top_logits[i, j]
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

                pred_instance = SinglePredict(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    has_score=has_score,
                    no_score=cur_not_answerable_score,
                    cls_score=pos_cls_score,
                )
                start_end.append(pred_instance)

    return start_end


def inference(args, qa_model, features, dataset_processor):
    """
    Model inference during validation or final evaluation.
    """
    ctx_l = parse_ctx(args.gpus)
    # We process all the chunk features
    all_chunk_features = []
    chunk_feature_ptr = [0]
    for feature in features:
        chunk_features = dataset_processor.process_sample(feature)
        all_chunk_features.extend(chunk_features)
        chunk_feature_ptr.append(chunk_feature_ptr[-1] + len(chunk_features))

    # re-generate the dataloader
    dataloader = mx.gluon.data.DataLoader(
        all_chunk_features,
        batchify_fn=dataset_processor.BatchifyFunction,
        batch_size=args.eval_batch_size,
        num_workers=0,
        shuffle=False)

    all_results = []
    epoch_tic = time.time()
    tic = time.time()
    total_num = 0
    log_interval = 10
    epoch_size = len(features)
    for batch_idx, batch in enumerate(grouper(dataloader, len(ctx_l))):
        # Predict for each chunk
        for sample, ctx in zip(batch, ctx_l):
            if sample is None:
                continue
            # Copy the data to device
            tokens = sample.data.as_in_ctx(ctx)
            total_num += len(tokens)
            segment_ids = sample.segment_ids.as_in_ctx(ctx)
            valid_length = sample.valid_length.as_in_ctx(ctx)
            p_mask = sample.masks.as_in_ctx(ctx)
            p_mask = 1 - p_mask  # In the network, we use 1 --> no_mask, 0 --> mask
            start_top_logits, start_top_index, end_top_logits, end_top_index, answerable_logits, \
                pos_cls_logits = qa_model.inference(tokens, segment_ids, valid_length, p_mask,
                                                    args.start_top_n, args.end_top_n)
            for i, qas_id in enumerate(sample.qas_id):
                result = RawResultExtended(qas_id=qas_id,
                                           start_top_logits=start_top_logits[i].asnumpy(),
                                           start_top_index=start_top_index[i].asnumpy(),
                                           end_top_logits=end_top_logits[i].asnumpy(),
                                           end_top_index=end_top_index[i].asnumpy(),
                                           answerable_logits=answerable_logits[i].asnumpy(),
                                           pos_cls_logits=pos_cls_logits[i].asnumpy())

                all_results.append(result)

        # logging
        if (batch_idx + 1)  % log_interval == 0:
            # Output the loss of per step
            toc = time.time()
            logging.info(
                '[batch {}], Time cost={:.2f},'
                ' Throughput={:.2f} samples/s, ETA={:.2f}h'.format(
                    batch_idx + 1, toc - tic, total_num / (toc - tic),
                    (epoch_size - total_num) / (total_num / (toc - epoch_tic)) / 3600))
            tic = time.time()

    epoch_toc = time.time()
    logging.info('Time cost=%2f s, Thoughput=%.2f samples/s', epoch_toc - epoch_tic,
                 total_num / (epoch_toc - epoch_tic))


    infer_features = collections.OrderedDict()
    for index, (left_index, right_index) in enumerate(zip(chunk_feature_ptr[:-1],
                                                          chunk_feature_ptr[1:])):
        chunked_features = all_chunk_features[left_index:right_index]
        results = all_results[left_index:right_index]
        original_feature = features[index]
        qas_ids = set([result.qas_id for result in results] +
                      [feature.qas_id for feature in chunked_features])
        assert len(qas_ids) == 1, 'Mismatch Occured between features and results'
        example_qas_id = list(qas_ids)[0]
        assert example_qas_id == original_feature.qas_id, \
            'Mismatch Occured between original feature and chunked features'
        start_ends = predict_extended(
            feature=original_feature,
            chunked_features=chunked_features,
            results=results,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            start_top_n=args.start_top_n,
            end_top_n=args.end_top_n)
        infer_features[example_qas_id] = start_ends

    return infer_features


def ensemble(args, is_save=True):
    ctx_l = parse_ctx(args.gpus)
    cfg, tokenizer, qa_net = get_network(
        args.model_name, args.classifier_dropout)
    if not args.voter_path:
        # prepare train set
        train_cache_path = os.path.join(
            CACHE_PATH, 'train_{}_squad_{}.ndjson'.format(
                args.model_name, args.version))
        if os.path.exists(train_cache_path) and not args.overwrite_cache:
            train_features = []
            with open(train_cache_path, 'r') as f:
                for line in f:
                    train_features.append(SquadFeature.from_json(line))
            logging.info('Found cached training features, load from {}'.format(train_cache_path))

        else:
            # Load the data
            logging.info('Load data from {}, Version={}'.format(args.data_dir, args.version))
            train_examples = get_squad_examples(args.data_dir, segment='train', version=args.version)
            start = time.time()
            num_process = min(cpu_count(), 8)
            logging.info('Tokenize Training Data:')
            with Pool(num_process) as pool:
                train_features = pool.map(functools.partial(convert_squad_example_to_feature,
                                                            tokenizer=tokenizer,
                                                            is_training=False), train_examples)
            logging.info('Done! Time spent:{}'.format(time.time() - start))
            with open(train_cache_path, 'w') as f:
                for feature in train_features:
                    f.write(feature.to_json() + '\n')
        train_data_path = os.path.join(args.data_dir, 'train-v{}.json'.format(args.version))

    dataset_processor = SquadDatasetProcessor(tokenizer=tokenizer,
                                              doc_stride=args.doc_stride,
                                              max_seq_length=args.max_seq_length,
                                              max_query_length=args.max_query_length)
    # prepare dev set
    dev_cache_path = os.path.join(CACHE_PATH,
                                  'dev_{}_squad_{}.ndjson'.format(args.model_name,
                                                                  args.version))
    if os.path.exists(dev_cache_path) and not args.overwrite_cache:
        dev_features = []
        with open(dev_cache_path, 'r') as f:
            for line in f:
                dev_features.append(SquadFeature.from_json(line))
        logging.info('Found cached dev features, load from {}'.format(dev_cache_path))
    else:
        logging.info('Load data from {}, Version={}'.format(args.data_dir, args.version))
        dev_examples = get_squad_examples(args.data_dir, segment='dev', version=args.version)
        start = time.time()
        num_process = min(cpu_count(), 8)
        logging.info('Tokenize Dev Data:')
        with Pool(num_process) as pool:
            dev_features = pool.map(functools.partial(convert_squad_example_to_feature,
                                                      tokenizer=tokenizer,
                                                      is_training=False), dev_examples)
        logging.info('Done! Time spent:{}'.format(time.time() - start))
        with open(dev_cache_path, 'w') as f:
            for feature in dev_features:
                f.write(feature.to_json() + '\n')

    dev_data_path = os.path.join(args.data_dir, 'dev-v{}.json'.format(args.version))

    def inference_single_ckpt(param_checkpoint, all_start_ends, features):
        logging.info('Inference the fine-tuned parameters {}'.format(param_checkpoint))
        qa_net.load_parameters(
            param_checkpoint,
            ctx=ctx_l,
            cast_dtype=True,
            ignore_extra=True,
            allow_missing=True)
        cur_features = inference(args, qa_net, features, dataset_processor)

        # update all_start_ends
        for qas_id, start_ends in cur_features.items():
            if qas_id not in all_start_ends:
                all_start_ends[qas_id] = start_ends
            else:
                all_start_ends[qas_id].extend(start_ends)

    def scatter_and_update(all_start_ends, factor=1.0):
        all_predictions = {}
        all_scores = {}
        for qas_id, start_end_list in all_start_ends.items():
            has_ans_dict = collections.OrderedDict()
            no_score, cls_score = 10000000, 10000000
            for instance in start_end_list:
                start = instance.start_idx
                end = instance.end_idx
                has_score = instance.has_score
                cls_score = min(instance.cls_score, cls_score)
                no_score = min(instance.no_score, no_score)

                if (start, end) not in has_ans_dict:
                    has_ans_dict[(start, end)] = [has_score]
                else:
                    has_ans_dict[(start, end)].append(has_score)

            for pos, scores in has_ans_dict.items():
                # (start_idx, end_idx), has_score
                has_ans_dict[pos] = np.sum(scores) / float(factor)

            # Get the the start and end positions with highest has_score
            if len(has_ans_dict):
                (start_idx, end_idx), highest_has_score = has_ans_dict.popitem(0)
            else:
                # There is no valid after inference
                start_idx, end_idx = 0, 0
                highest_has_score, no_score, cls_score = -10000000, 10000000, 10000000
            all_predictions[qas_id] = (start_idx, end_idx)
            all_scores[qas_id] = [highest_has_score, no_score, cls_score]

        return all_predictions, all_scores

    filenames = [
        os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if '.params' in f]
    filenames.sort(key=lambda ele: (len(ele), ele), reverse=True)
    # TODO(zheyuye), weight scheming with different values of each ckpt
    # train a machine learning based voter
    if not args.voter_path:
        train_start_ends = {}
        inference_single_ckpt(filenames[0], train_start_ends, train_features)
        _, train_scores = scatter_and_update(train_start_ends)
        ml_voter_path = os.path.join(args.output_dir, 'voter.params')
        ml_voter(train_scores, ml_voter_path, data_file=train_data_path, is_training=True)
        training_scores_file = os.path.join(args.output_dir, 'training_scores.json')
        with open(training_scores_file, 'w') as of:
            of.write(json.dumps(train_scores, indent=4) + '\n')
    else:
        ml_voter_path = args.voter_path

    dev_start_ends = {}
    for idx, param_checkpoint in enumerate(filenames):
        inference_single_ckpt(param_checkpoint, dev_start_ends, dev_features)

        # Update the results step step
        all_predictions, all_scores = scatter_and_update(dev_start_ends, idx + 1)
        dev_scores_file = os.path.join(args.output_dir, 'dev_scores-{}.json'.format())
        with open(dev_scores_file, 'w') as of:
            of.write(json.dumps(all_scores, indent=4) + '\n')
        no_answer_score_json = ml_voter(all_scores, ml_voter_path, is_training=False)
        # make the predictions
        for feature in dev_features:
            context_text = feature.context_text
            context_token_offsets = feature.context_token_offsets
            qas_id = feature.qas_id
            start_idx, end_idx = all_predictions[qas_id]
            pred_answer = context_text[context_token_offsets[start_idx][0]:
                                       context_token_offsets[end_idx][1]]
            all_predictions[qas_id] = pred_answer

        na_prob = no_answer_score_json if args.version == '2.0' else None
        eval_dict, revised_result = squad_eval(dev_data_path, all_predictions, na_prob, revise=True)

    if is_save:
        logging.info('The evaluated files are saved in {}'.format(args.output_dir))
        output_prediction_file = os.path.join(args.output_dir, 'predictions.json')
        na_prob_file = os.path.join(args.output_dir, 'na_prob.json')
        revised_prediction_file = os.path.join(args.output_dir, 'revised_predictions.json')

        with open(output_prediction_file, 'w') as of:
            of.write(json.dumps(all_predictions, indent=4) + '\n')
        with open(na_prob_file, 'w') as of:
            of.write(json.dumps(no_answer_score_json, indent=4) + '\n')
        with open(revised_prediction_file, 'w') as of:
            of.write(json.dumps(revised_result, indent=4) + '\n')

    logging.info('The evaluated results are {}'.format(json.dumps(eval_dict)))
    output_eval_results_file = os.path.join(args.output_dir, 'results.json')
    with open(output_eval_results_file, 'w') as of:
        of.write(json.dumps(eval_dict, indent=4) + '\n')


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    os.environ['MXNET_USE_FUSION'] = '0'  # Manually disable pointwise fusion
    args = parse_args()
    logging_config(args.output_dir, name='ensemble_squad{}'.format(args.version))
    set_seed(args.seed)
    if args.do_ensemble:
        ensemble(args)
