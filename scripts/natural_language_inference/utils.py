"""
utils.py

Part of NLI scripts of gluon-nlp.
Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>
"""

from mxnet import nd
import gluonnlp as nlp
import nltk

def fetch_embedding_of_sentence(sentence: str, embedding: nlp.embedding.TokenEmbedding):
    """
    Fetch embeddings of the given sequence. Tokenizing with NLTK.
    Args:
        sentence: the sentence to process
        embedding: TokenEmbedding object that store the embeddings look-up table
    Return:
        A ndarray of embeddings.
    """
    words = nltk.word_tokenize(sentence.lower())
    embeds = [embedding[w] for w in words]
    # normalize
    # embeds = [t / (t**2).sum().sqrt() * 0.05 for t in embeds]
    return nd.stack(*embeds)

def pad_sentences(sentences):
    """
    Padding the sentences in one minibatch to the longest length in this minibatch.
    Args:
        sentences: list of ndarray, which represents the input sentences
    Return:
        Padded ndarray
    """
    length = 0
    dim = sentences[0].shape[1]
    for item in sentences:
        if item.shape[0] > length:
            length = item.shape[0]
    ret = []
    for item in sentences:
        if length > item.shape[0]:
            ret.append(
                nd.concat(item, nd.zeros((length-item.shape[0], dim)), dim=0))
        else:
            ret.append(item)
    return nd.stack(*ret)
