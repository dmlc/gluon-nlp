#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
set -e

# Glove
for i in glove.42B.300d glove.6B.100d glove.6B.200d glove.6B.300d glove.6B.50d glove.840B.300d glove.twitter.27B.100d glove.twitter.27B.200d glove.twitter.27B.25d glove.twitter.27B.50d
do
    echo "Running $i"
    python evaluate_pretrained.py --gpu 0  --embedding-name glove --embedding-source $i --logdir results --max-vocab-size 300000 --analogy-datasets GoogleAnalogyTestSet BiggerAnalogyTestSet
done

# Fasttext
for i in crawl-300d-2M wiki-news-300d-1M wiki-news-300d-1M-subword
do
    echo "Running $i"
    python evaluate_pretrained.py --gpu 0  --embedding-name fasttext --embedding-source $i --logdir results --max-vocab-size 300000 --analogy-datasets GoogleAnalogyTestSet BiggerAnalogyTestSet
done

# Fasttext with subwords
for i in wiki.en wiki.simple
do
    echo "Running $i"
    python evaluate_pretrained.py --gpu 0  --embedding-name fasttext --fasttext-load-ngrams --embedding-source $i --logdir results --max-vocab-size 300000 --analogy-datasets GoogleAnalogyTestSet BiggerAnalogyTestSet
done
