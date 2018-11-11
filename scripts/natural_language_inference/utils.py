"""
utils.py

Part of NLI scripts of gluon-nlp.
Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>
"""

import gluonnlp as nlp
import nltk

def tokenize_and_index(sentence: str, embedding: nlp.embedding.TokenEmbedding):
    """
    Tokenize sentence with NLTK and obtain token index in provided embedding object.
    """
    words = nltk.word_tokenize(sentence.lower())
    return [embedding.token_to_idx[t] for t in words]
