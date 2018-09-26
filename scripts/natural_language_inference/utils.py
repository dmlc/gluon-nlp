import mxnet as mx
from mxnet import nd
import gluonnlp as nlp
import nltk

def fetch_embedding_of_sentence(sentence: str, embedding: nlp.embedding.TokenEmbedding):
    words = nltk.word_tokenize(sentence.lower())
    embeds = [embedding[w] for w in words]
    return nd.stack(*embeds)

def pad_sentences(sentences):
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

