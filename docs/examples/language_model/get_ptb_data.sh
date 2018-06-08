#!/usr/bin/env bash

curl -O http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar --strip-components=3 -xzvf simple-examples.tgz ./simple-examples/data/ptb.train.txt ./simple-examples/data/ptb.valid.txt ./simple-examples/data/ptb.test.txt
