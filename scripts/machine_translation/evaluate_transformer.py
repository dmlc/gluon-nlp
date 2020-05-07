import numpy as np
import random
import os
import mxnet as mx
from mxnet import gluon
import argparse
import logging
import io
import time
from gluonnlp.utils.misc import logging_config
from gluonnlp.models.transformer import TransformerNMTModel,\
    TransformerNMTInference
from gluonnlp.data.batchify import Tuple, Pad, Stack
from gluonnlp.data.filtering import MosesNormalizer
from gluonnlp.data import tokenizers
from gluonnlp.sequence_sampler import BeamSearchSampler, BeamSearchScorer
import sacrebleu
mx.npx.set_np()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Transformer for Neural Machine Translation. Load a checkpoint and inference.')
    parser.add_argument('--seed', type=int, default=100, help='The random seed.')
    parser.add_argument('--src_lang', type=str, default='en', help='Source language')
    parser.add_argument('--tgt_lang', type=str, default='de', help='Target language')
    parser.add_argument('--src_corpus', type=str,
                        help='The source corpus for evaluation.')
    parser.add_argument('--tgt_corpus', type=str,
                        help='The target corpus for evaluation.')
    parser.add_argument('--src_tokenizer', choices=['spm',
                                                    'subword_nmt',
                                                    'yttm',
                                                    'hf_bytebpe',
                                                    'hf_wordpiece',
                                                    'hf_bpe'],
                        required=True, type=str,
                        help='The source tokenizer. Only supports online encoding at present.')
    parser.add_argument('--tgt_tokenizer', choices=['spm',
                                                    'subword_nmt',
                                                    'yttm',
                                                    'hf_bytebpe',
                                                    'hf_wordpiece',
                                                    'hf_bpe'],
                        required=True, type=str,
                        help='The target tokenizer. Only supports online encoding at present.')    
    parser.add_argument('--src_subword_model_path', type=str,
                        help='Path to the source subword model.')
    parser.add_argument('--src_vocab_path', type=str,
                        help='Path to the source subword vocab.')
    parser.add_argument('--tgt_subword_model_path', type=str,
                        help='Path to the target subword model.')
    parser.add_argument('--tgt_vocab_path', type=str,
                        help='Path to the target subword vocab.')
    parser.add_argument('--src_max_len', type=int, default=None,
                        help='Maximum length of the source sentence.')
    parser.add_argument('--tgt_max_len', type=int, default=None,
                        help='Maximum length of the target sentence.')
    parser.add_argument('--cfg', type=str, help='Config file of the Transformer model.')
    parser.add_argument('--beam-size', type=int, default=4, help='Number of beams')
    parser.add_argument('--lp_alpha', type=float, default=0.6,
                        help='The alpha value in the length penalty')
    parser.add_argument('--lp_k', type=int, default=5, help='The K value in the length penalty')
    parser.add_argument('--max_length_a', type=int, default=1,
                        help='The a in the a * x + b formula of beam search')
    parser.add_argument('--max_length_b', type=int, default=50,
                        help='The b in the a * x + b formula of beam search')
    parser.add_argument('--param_path', type=str, help='The path to the model parameters.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.'
                             '(using single gpu is suggested)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='The path to save the log files and predictions.')
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.splitext(args.param_path)[0] + '_evaluation'
    logging_config(args.save_dir, console=True)
    logging.info(args)
    return args


def process_corpus(corpus_path, sentence_normalizer, bpe_tokenizer,
                   base_tokenizer=None, add_bos=True, add_eos=True):
    processed_token_ids = []
    raw_lines = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw_lines.append(line)
            line = line.strip()
            line = sentence_normalizer(line)
            if base_tokenizer is not None:
                line = ' '.join(base_tokenizer.encode(line))
            bpe_token_ids = bpe_tokenizer.encode(line, output_type=int)
            if add_bos:
                bpe_token_ids = [bpe_tokenizer.vocab.bos_id] + bpe_token_ids
            if add_eos:
                bpe_token_ids.append(bpe_tokenizer.vocab.eos_id)
            processed_token_ids.append(bpe_token_ids)
    return processed_token_ids, raw_lines


def create_tokenizer(tokenizer_type, model_path, vocab_path):
    if tokenizer_type == 'spm':
        return tokenizers.create(tokenizer_type, model_path=model_path, vocab=vocab_path)
    elif tokenizer_type == 'subword_nmt':
        return tokenizers.create(tokenizer_type, codec_path=model_path, vocab_path=vocab_path)
    elif tokenizer_type == 'yttm':
        return tokenizers.create(tokenizer_type, model_path=model_path)
    elif tokenizer_type == 'hf_bytebpe':
        return tokenizers.create(tokenizer_type, merges_file=model_path, vocab_file=vocab_path)
    elif tokenizer_type == 'hf_wordpiece':
        return tokenizers.create(tokenizer_type, vocab_file=vocab_path)
    elif tokenizer_type == 'hf_bpe':
        return tokenizers.create(tokenizer_type, merges_file=model_path, vocab_file=vocab_path)
    else:
        raise NotImplementedError


def evaluate(args):
    ctx_l = [mx.cpu()] if args.gpus is None or args.gpus == '' else [mx.gpu(int(x)) for x in
                                                                     args.gpus.split(',')]
    src_normalizer = MosesNormalizer(args.src_lang)
    tgt_normalizer = MosesNormalizer(args.tgt_lang)
    base_src_tokenizer = tokenizers.create('moses', args.src_lang)
    base_tgt_tokenizer = tokenizers.create('moses', args.tgt_lang)

    src_tokenizer = create_tokenizer(args.src_tokenizer,
                                     args.src_subword_model_path,
                                     args.src_vocab_path)
    tgt_tokenizer = create_tokenizer(args.tgt_tokenizer,
                                     args.tgt_subword_model_path,
                                     args.tgt_vocab_path)
    src_vocab = src_tokenizer.vocab
    tgt_vocab = tgt_tokenizer.vocab
    cfg = TransformerNMTModel.get_cfg().clone_merge(args.cfg)
    cfg.defrost()
    cfg.MODEL.src_vocab_size = len(src_vocab)
    cfg.MODEL.tgt_vocab_size = len(tgt_vocab)
    cfg.freeze()
    model = TransformerNMTModel.from_cfg(cfg)
    model.hybridize()
    model.load_parameters(args.param_path, ctx=ctx_l)
    inference_model = TransformerNMTInference(model=model)
    inference_model.hybridize()
    # Construct the BeamSearchSampler
    beam_search_sampler = BeamSearchSampler(beam_size=args.beam_size,
                                            decoder=inference_model,
                                            vocab_size=len(tgt_vocab),
                                            eos_id=tgt_vocab.eos_id,
                                            scorer=BeamSearchScorer(alpha=args.lp_alpha,
                                                                    K=args.lp_k,
                                                                    from_logits=False),
                                            max_length_a=args.max_length_a,
                                            max_length_b=args.max_length_b)
    logging.info(beam_search_sampler)
    ctx = ctx_l[0]
    avg_nll_loss = 0
    ntokens = 0
    pred_sentences = []
    start_eval_time = time.time()
    all_src_token_ids, all_src_lines = process_corpus(args.src_corpus,
                                                      sentence_normalizer=src_normalizer,
                                                      base_tokenizer=base_src_tokenizer,
                                                      bpe_tokenizer=src_tokenizer,
                                                      add_bos=False,
                                                      add_eos=True)
    all_tgt_token_ids, all_tgt_lines = process_corpus(args.tgt_corpus,
                                                      sentence_normalizer=tgt_normalizer,
                                                      base_tokenizer=base_tgt_tokenizer,
                                                      bpe_tokenizer=tgt_tokenizer,
                                                      add_bos=True,
                                                      add_eos=True)
    test_dataloader = gluon.data.DataLoader(
        list(zip(all_src_token_ids,
                 [len(ele) for ele in all_src_token_ids],
                 all_tgt_token_ids,
                 [len(ele) for ele in all_tgt_token_ids])),
        batch_size=32,
        batchify_fn=Tuple(Pad(), Stack(), Pad(), Stack()),
        shuffle=False)
    for i, (src_token_ids, src_valid_length, tgt_token_ids, tgt_valid_length)\
            in enumerate(test_dataloader):
        src_token_ids = mx.np.array(src_token_ids, ctx=ctx, dtype=np.int32)
        src_valid_length = mx.np.array(src_valid_length, ctx=ctx, dtype=np.int32)
        tgt_token_ids = mx.np.array(tgt_token_ids, ctx=ctx, dtype=np.int32)
        tgt_valid_length = mx.np.array(tgt_valid_length, ctx=ctx, dtype=np.int32)
        tgt_pred = model(src_token_ids, src_valid_length, tgt_token_ids[:, :-1],
                         tgt_valid_length - 1)
        pred_logits = mx.npx.log_softmax(tgt_pred, axis=-1)
        nll = - mx.npx.pick(pred_logits, tgt_token_ids[:, 1:])
        avg_nll_loss += mx.npx.sequence_mask(nll,
                                             sequence_length=tgt_valid_length - 1,
                                             use_sequence_length=True, axis=1).sum().asnumpy()
        ntokens += int((tgt_valid_length - 1).sum().asnumpy())
        init_input = mx.np.array([tgt_vocab.bos_id for _ in range(src_token_ids.shape[0])], ctx=ctx)
        states = inference_model.init_states(src_token_ids, src_valid_length)
        samples, scores, valid_length = beam_search_sampler(init_input, states, src_valid_length)
        for j in range(samples.shape[0]):
            pred_tok_ids = samples[j, 0, :valid_length[j, 0].asnumpy()].asnumpy().tolist()
            bpe_decode_line = tgt_tokenizer.decode(pred_tok_ids[1:-1])
            pred_sentence = base_tgt_tokenizer.decode(bpe_decode_line.split(' '))
            pred_sentences.append(pred_sentence)
            print(pred_sentence)
        print('Processed {}/{}'.format(len(pred_sentences), len(all_tgt_lines)))
    end_eval_time = time.time()
    avg_nll_loss = avg_nll_loss / ntokens

    with io.open(os.path.join(args.save_dir, 'gt_sentences.txt'), 'w', encoding='utf-8') as of:
        for line in all_tgt_lines:
            of.write(line + '\n')
    with io.open(os.path.join(args.save_dir, 'pred_sentences.txt'), 'w', encoding='utf-8') as of:
        for line in pred_sentences:
            of.write(line + '\n')

    sacrebleu_out = sacrebleu.corpus_bleu(sys_stream=pred_sentences, ref_streams=[all_tgt_lines])
    logging.info('Time Spent: {}, #Sent={}, SacreBlEU={} Avg NLL={}, Perplexity={}'
                 .format(end_eval_time - start_eval_time, len(all_tgt_lines),
                         sacrebleu_out.score, avg_nll_loss, np.exp(avg_nll_loss)))


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    os.environ['MXNET_USE_FUSION'] = '0'  # Manually disable pointwise fusion
    args = parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    random.seed(args.seed)
    evaluate(args)
