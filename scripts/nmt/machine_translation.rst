Machine Translation
-------------------

:download:`[Download] </scripts/nmt.zip>`

Use the following command to train the GNMT model on the IWSLT2015 dataset.

.. code-block:: console

   $ python train_gnmt.py --src_lang en --tgt_lang vi --batch_size 64 \
                   --optimizer adam --lr 0.001 --lr_update_factor 0.5 --beam_size 10 \
                   --num_hidden 512 --save_dir gnmt_en_vi_l2_h512_beam10 --epochs 10 --gpu 0

It gets test BLEU score equals to 26.22.

Use the following command to train the Transformer model on the WMT16 dataset for English to German translation. 

.. code-block:: console

   $ python train_transformer.py --src_lang en --tgt_lang de --batch_size 4096 \
                          --optimizer adam --num_accumulated 8 --lr 1.0 --warmup_steps 8000 \
                          --save_dir transformer_en_de_u512 --epochs 40 --gpus 0,1,2,3
                          --average_checkpoint True --num_averages 5 --num_buckets 20

It gets test BLEU score equals to 27.79 on newstest2014.
