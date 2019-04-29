import mxnet as mx
import gluonnlp as nlp
import argparse
import pickle
from collections import namedtuple


BERTModelMetadata = namedtuple('BERTModelMetadata', ['config', 'tag_vocab'])


def _metadata_file_path(checkpoint_prefix):
    return checkpoint_prefix + "_metadata.pkl"


def dump_metadata(config, tag_vocab):
    metadata = BERTModelMetadata(config=config, tag_vocab=tag_vocab)
    with open(_metadata_file_path(config.save_checkpoint_prefix), "wb") as ofp:
        pickle.dump(metadata, ofp)


def load_metadata(checkpoint_prefix):
    with open(_metadata_file_path(checkpoint_prefix), "rb") as ifp:
        metadata = pickle.load(ifp)
        return metadata.config, metadata.tag_vocab


def get_context(gpu_index):
    """ This method gets context of execution"""
    context = None
    if gpu_index is None or gpu_index == '':
        context = mx.cpu()
    if isinstance(gpu_index, int):
        context = mx.gpu(gpu_index)
    return context


def str2bool(v: str):
    """Utility function for parsing boolean in argparse

    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    :param v: value of the argument
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_bert_dataset_name(is_cased):
    """Returns relevant BERT dataset name, depending on whether we are using a cased model.

    Parameters
    ----------
    is_cased: bool
        Whether we are using a cased model.

    Returns
    -------
    str: Named of the BERT dataset.

    """
    if is_cased:
        return 'book_corpus_wiki_en_cased'
    else:
        return 'book_corpus_wiki_en_uncased'


def get_bert_model(bert_model, cased, ctx, dropout_prob):
    bert_dataset_name = get_bert_dataset_name(cased)

    return nlp.model.get_bert_model(
        model_name=bert_model,
        dataset_name=bert_dataset_name,
        pretrained=True,
        ctx=ctx,
        use_pooler=False,
        use_decoder=False,
        use_classifier=False,
        dropout=dropout_prob,
        embed_dropout=dropout_prob)


