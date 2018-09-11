Machine Translation
-------------------

:download:`[Download] </scripts/nmt.zip>`

Use the following command to train the GNMT model on the IWSLT2015 dataset.

.. code-block:: console

   $ MXNET_GPU_MEM_POOL_TYPE=Round python train_gnmt.py --src_lang en --tgt_lang vi --batch_size 128 \
                   --optimizer adam --lr 0.001 --lr_update_factor 0.5 --beam_size 10 --bucket_scheme exp \
                   --num_hidden 512 --save_dir gnmt_en_vi_l2_h512_beam10 --epochs 12 --gpu 0

It gets test BLEU score equals to 26.20.

Use the following commands to train the Transformer model on the WMT14 dataset for English to German translation.

.. code-block:: console

   $ MXNET_GPU_MEM_POOL_TYPE=Round python train_transformer.py --dataset WMT2014BPE \
                          --src_lang en --tgt_lang de --batch_size 2700 \
                          --optimizer adam --num_accumulated 16 --lr 2.0 --warmup_steps 4000 \
                          --save_dir transformer_en_de_u512 --epochs 30 --gpus 0,1,2,3,4,5,6,7 --scaled \
                          --average_start 5 --num_buckets 20 --bucket_scheme exp --bleu 13a --log_interval 10

It gets official mteval-v13a BLEU score equals to 27.09 on newstest2014 (http://statmt.org/wmt14/test-filtered.tgz).
This result is obtained by using averaged SGD in last 5 epochs. If we use international tokenization (i.e., ``--bleu intl``),
we can obtain bleu score equals to 27.89. If we use ``--bleu tweaked``, we obtain test BLEU score equals to 28.96.
This result is obtained on tweaked reference, where the tokenized reference text is put in ATAT format for historical reason
and following preprocessing pipeline is done:

.. code-block:: console

    mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l de
    mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl
    mosesdecoder/scripts/tokenizer/tokenizer.perl -q -no-escape -protected mosesdecoder/scripts/tokenizer/basic-protected-patterns -l de.

If we turn on  ``--full``, the testing is performed on newstest2014 (http://statmt.org/wmt14/test-full.tgz). Then, we can
obtain BLEU=27.05 with ``--bleu 13a``, BLEU=27.81 with ``--bleu intl``, and BLEU=28.80 with ``--bleu tweaked``

The pre-trained model can be downloaded from http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/transformer_en_de_512_WMT2014-14bd361b.zip.

For the users from China, it might be faster with this link instead: https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/models/transformer_en_de_512_WMT2014-14bd361b.zip.
