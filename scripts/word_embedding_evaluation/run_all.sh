#!/bin/bash

# Glove
for i in glove.42B.300d glove.6B.100d glove.6B.200d glove.6B.300d glove.6B.50d glove.840B.300d glove.twitter.27B.100d glove.twitter.27B.200d glove.twitter.27B.25d glove.twitter.27B.50d
do
    echo "Running $i"
    python word_embedding_evaluation.py --gpu 0  --embedding-name glove --embedding-source $i --log results-vocablimit.csv --analogy-max-vocab 300000
done

# Fasttext
for i in wiki.en wiki.simple crawl-300d-2M wiki-news-300d-1M wiki-news-300d-1M-subword
do
    echo "Running $i"
    python word_embedding_evaluation.py --gpu 0  --embedding-name fasttext --embedding-source $i --log results-vocablimit.csv --analogy-max-vocab 300000
done
