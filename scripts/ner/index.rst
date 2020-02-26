Named Entity Recognition
------------------------

:download:`Download scripts </model_zoo/ner.zip>`

Reference: Devlin, Jacob, et al. "`Bert: Pre-training of deep bidirectional transformers for language understanding. <https://arxiv.org/abs/1810.04805>`_" arXiv preprint arXiv:1810.04805 (2018).

Named Entity Recognition with BERT 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GluonNLP provides training and prediction script for named entity recognition models.

The training script for NER requires the seqeval package:

.. code-block:: console

    $ pip install seqeval --user

Dataset should be formatted in `CoNLL-2003 shared task format <https://www.clips.uantwerpen.be/conll2003/ner/>`_.
Assuming data files are located in `${DATA_DIR}`, below command trains BERT model for
named entity recognition, and saves model artifacts to `${MODEL_DIR}` with `large_bert`
prefix in file names (assuming `${MODEL_DIR}` exists):

.. code-block:: console

    $ python finetune_bert.py \
        --train-path ${DATA_DIR}/train.txt \
        --dev-path ${DATA_DIR}/dev.txt \
        --test-path ${DATA_DIR}/test.txt \
        --gpu 0 --learning-rate 1e-5 --dropout-prob 0.1 --num-epochs 100 --batch-size 8 \
        --optimizer bertadam --bert-model bert_24_1024_16 \
        --save-checkpoint-prefix ${MODEL_DIR}/large_bert --seed 13531

This achieves Test F1 from `91.5` to `92.2` (`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/bert/finetuned_conll2003.log>`_).
