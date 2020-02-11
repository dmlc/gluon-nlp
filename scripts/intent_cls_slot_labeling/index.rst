Intent Classification and Slot Labeling
---------------------------------------

:download:`Download scripts </model_zoo/intent_cls_slot_labeling.zip>`

Reference:
- Devlin, Jacob, et al. "`Bert: Pre-training of deep bidirectional transformers for language understanding. <https://arxiv.org/abs/1810.04805>`_" arXiv preprint arXiv:1810.04805 (2018).
- Chen, Qian, et al. "`BERT for Joint Intent Classification and Slot Filling. <https://arxiv.org/pdf/1902.10909.pdf>`_"  arXiv preprint arXiv:1902.10909 (2019).

Joint Intent Classification and Slot Labelling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Intent classification and slot labelling are two essential problems in Natural Language Understanding (NLU).
In *intent classification*, the agent needs to detect the intention that the speaker's utterance conveys. For example, when the speaker says "Book a flight from Long Beach to Seattle", the intention is to book a flight ticket.
In *slot labelling*, the agent needs to extract the semantic entities that are related to the intent. In our previous example,
"Long Beach" and "Seattle" are two semantic constituents related to the flight, i.e., the origin and the destination.

Essentially, *intent classification* can be viewed as a sequence classification problem and *slot labelling* can be viewed as a
sequence tagging problem similar to Named-entity Recognition (NER). Due to their inner correlation, these two tasks are usually
trained jointly with a multi-task objective function.

Here's one example of the ATIS dataset, it uses the `IOB2 format <https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)>`__.

+-----------+--------------------------+--------------+
| Sentence  | Tags                     | Intent Label |
+===========+==========================+==============+
| are       | O                        | atis_flight  |
+-----------+--------------------------+--------------+
| there     | O                        |              |
+-----------+--------------------------+--------------+
| any       | O                        |              |
+-----------+--------------------------+--------------+
| flight    | O                        |              |
+-----------+--------------------------+--------------+
| from      | O                        |              |
+-----------+--------------------------+--------------+
| long      | B-fromloc.city_name      |              |
+-----------+--------------------------+--------------+
| beach     | I-fromloc.city_name      |              |
+-----------+--------------------------+--------------+
| to        | O                        |              |
+-----------+--------------------------+--------------+
| columbus  | B-toloc.city_name        |              |
+-----------+--------------------------+--------------+
| on        | O                        |              |
+-----------+--------------------------+--------------+
| wednesday | B-depart_date.day_name   |              |
+-----------+--------------------------+--------------+
| april     | B-depart_date.month_name |              |
+-----------+--------------------------+--------------+
| sixteen   | B-depart_date.day_number |              |
+-----------+--------------------------+--------------+



In this example, we demonstrate how to use GluonNLP to fine-tune a pretrained BERT model for joint intent classification and slot labelling. We
choose to finetune a pretrained BERT model.  We use two datasets `ATIS <https://github.com/yvchen/JointSLU>`__ and `SNIPS <https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines>`__.

The training script requires the seqeval and tqdm packages:

.. code-block:: console

    $ pip3 install seqeval --user
    $ pip3 install tqdm --user

For the ATIS dataset, use the following command to run the experiment:

.. code-block:: console

    $ python finetune_icsl.py --gpu 0 --dataset atis

It produces the final slot labelling F1 = `95.83%` and intent classification accuracy = `98.66%`

For the SNIPS dataset, use the following command to run the experiment:

.. code-block:: console

    $ python finetune_icsl.py --gpu 0 --dataset snips

It produces the final slot labelling F1 = `96.06%` and intent classification accuracy = `98.71%`

Also, we train the models with three random seeds and report the mean/std.

For ATIS

+--------------------------------------------------------------------------------------------+----------------+-------------+
|                                             Models                                         | Intent Acc (%) | Slot F1 (%) |
+============================================================================================+================+=============+
| `Intent Gating & self-attention, EMNLP 2018 <https://www.aclweb.org/anthology/D18-1417>`__ |    98.77       |  96.52      |
+--------------------------------------------------------------------------------------------+----------------+-------------+
| `BLSTM-CRF + ELMo, AAAI 2019, <https://arxiv.org/abs/1811.05370>`__                        |    97.42       |  95.62      |
+--------------------------------------------------------------------------------------------+----------------+-------------+
| `Joint BERT, Arxiv 2019, <https://arxiv.org/pdf/1902.10909.pdf>`__                         |    97.5        |  96.1       |
+--------------------------------------------------------------------------------------------+----------------+-------------+
| Ours                                                                                       |    98.66±0.00  |  95.88±0.04 |
+--------------------------------------------------------------------------------------------+----------------+-------------+

For SNIPS

+--------------------------------------------------------------------+----------------+-------------+
|                                   Models                           | Intent Acc (%) | Slot F1 (%) |
+====================================================================+================+=============+
| `BLSTM-CRF + ELMo, AAAI 2019 <https://arxiv.org/abs/1811.05370>`__ | 99.29          | 93.90       |
+--------------------------------------------------------------------+----------------+-------------+
| `Joint BERT, Arxiv 2019 <https://arxiv.org/pdf/1902.10909.pdf>`__  | 98.60          | 97.00       |
+--------------------------------------------------------------------+----------------+-------------+
| Ours                                                               | 98.81±0.13     | 95.94±0.10  |
+--------------------------------------------------------------------+----------------+-------------+
