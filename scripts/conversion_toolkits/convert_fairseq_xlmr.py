import os
import copy
import logging
import argparse

import mxnet as mx

from gluonnlp.utils.misc import logging_config
from gluonnlp.models.xlmr import XLMRModel, XLMRForMLM
from gluonnlp.third_party import sentencepiece_model_pb2
from fairseq.models.roberta import XLMRModel as fairseq_XLMRModel
from convert_fairseq_roberta import rename, test_model, test_vocab, convert_config, convert_params
from gluonnlp.data.tokenizers import SentencepieceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the fairseq XLM-R Model to Gluon.')
    parser.add_argument('--fairseq_model_path', type=str, required=True,
                        help='Directory of the fairseq XLM-R model.')
    parser.add_argument('--model_size', type=str, choices=['base', 'large'], default='base',
                        help='Size of XLM-R model.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory path to save the converted XLM-R model.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='The single gpu to run mxnet, (e.g. --gpu 0) the default device is cpu.')
    parser.add_argument('--test', action='store_true',
                        help='Whether to test the conversion.')
    return parser.parse_args()

def convert_vocab(args, fairseq_model):
    print('converting vocab')
    origin_spm_path = os.path.join(args.fairseq_model_path, 'sentencepiece.bpe.model')
    assert os.path.exists(origin_spm_path)
    new_spm_path = os.path.join(args.save_dir, 'sentencepiece.model')
    fairseq_vocab = fairseq_model.task.dictionary
    # bos_word attr missing in fairseq_vocab
    fairseq_vocab.bos_word = fairseq_vocab[fairseq_vocab.bos_index]

    # model.pieces: <unk> <s> </s> other_tokens ->
    # model.pieces: <s> <pad> </s> <unk> other_tokens <mask>
    model = sentencepiece_model_pb2.ModelProto()
    with open(origin_spm_path, 'rb') as f_m:
        model.ParseFromString(f_m.read())
    p0 = model.pieces[0]
    p1 = model.pieces[1]
    p2 = model.pieces[2]

    pad_piece = copy.deepcopy(p0)
    pad_piece.piece = fairseq_vocab.pad_word
    pad_piece.type = pad_piece.CONTROL
    mask_piece = copy.deepcopy(p0)
    mask_piece.piece = '<mask>'
    mask_piece.type = mask_piece.CONTROL

    p0.type = p0.CONTROL
    p0.piece = fairseq_vocab.bos_word
    p1.type = p1.CONTROL
    p1.piece = fairseq_vocab.eos_word
    p2.type = p2.UNKNOWN
    p2.piece = fairseq_vocab.unk_word
    model.pieces.insert(fairseq_vocab.pad_index, pad_piece)
    model.pieces.append(mask_piece)

    model.trainer_spec.vocab_size = len(fairseq_vocab)
    model.trainer_spec.unk_id = fairseq_vocab.unk_index
    model.trainer_spec.bos_id = fairseq_vocab.bos_index
    model.trainer_spec.eos_id = fairseq_vocab.eos_index
    model.trainer_spec.pad_id = fairseq_vocab.pad_index

    with open(new_spm_path, 'wb') as f:
        f.write(model.SerializeToString())

    gluon_tokenizer = SentencepieceTokenizer(new_spm_path)
    if args.test:
        test_vocab(fairseq_model, gluon_tokenizer, check_all_tokens=True)

    vocab_size = len(fairseq_model.task.dictionary)
    print('| converted dictionary: {} types'.format(vocab_size))
    return vocab_size

def convert_fairseq_model(args):
    if not args.save_dir:
        args.save_dir = os.path.basename(args.fairseq_model_path) + '_gluon'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    fairseq_xlmr = fairseq_XLMRModel.from_pretrained(args.fairseq_model_path,
                                                     checkpoint_file='model.pt')
    vocab_size = convert_vocab(args, fairseq_xlmr)

    gluon_cfg = convert_config(fairseq_xlmr.args, vocab_size,
                               XLMRModel.get_cfg().clone())
    with open(os.path.join(args.save_dir, 'model.yml'), 'w') as of:
        of.write(gluon_cfg.dump())

    ctx = mx.gpu(args.gpu) if args.gpu is not None else mx.cpu()

    gluon_xlmr = convert_params(fairseq_xlmr,
                                   gluon_cfg,
                                   ctx)
    if args.test:
        test_model(fairseq_xlmr, gluon_xlmr, args.gpu)

    gluon_xlmr.save_parameters(os.path.join(args.save_dir, 'model_mlm.params'), deduplicate=True)
    logging.info('Convert the RoBERTa MLM model in {} to {}'.
                 format(os.path.join(args.fairseq_model_path, 'model.pt'), \
                        os.path.join(args.save_dir, 'model_mlm.params')))
    gluon_xlmr.backbone_model.save_parameters(
        os.path.join(args.save_dir, 'model.params'), deduplicate=True)
    logging.info('Convert the RoBERTa backbone model in {} to {}'.
                 format(os.path.join(args.fairseq_model_path, 'model.pt'), \
                        os.path.join(args.save_dir, 'model.params')))

    logging.info('Conversion finished!')
    logging.info('Statistics:')
    rename(args.save_dir)

if __name__ == '__main__':
    args = parse_args()
    logging_config()
    convert_fairseq_model(args)
