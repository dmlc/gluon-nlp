Machine Translation
-------------------

The script trains the GNMT model on the IWSLT2015 dataset.

.. code-block:: bash

   $ python gnmt.py --src_lang en --tgt_lang vi --batch_size 64 \
                    --optimizer adam --lr 0.001 --lr_update_factor 0.5 \
                    --num_hidden 512 --save_dir gnmt_en_vi_l2_h512_beam4 --epochs 80

It reaches test BLEU score equals 25.80 after 8 epochs.
