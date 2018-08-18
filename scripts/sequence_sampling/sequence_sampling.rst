Sequence Sampling
-----------------
:download:`[Download] </scripts/sequence_sampling.zip>`

This script can be used to generate sentences using beam search from a pre-trained language model.

Beam Search Generator
~~~~~~~~~~~~~~~~~~~~~

Use the following command to decode using beam search.

.. code-block:: console

   $ python sequence_sampling.py --use-beam-search --bos I love it --beam_size 5 --print_num 5

Output is

.. code-block:: console

   Beam Seach Parameters: beam_size=5, alpha=0.0, K=5
   Generation Result:
   [u'I love it .', -1.1241297]
   [u'I love it " .', -4.001592]
   [u'I love it , but it is not a <unk> .', -15.624882]
   [u'I love it , but it is not a <unk> , but it is not a <unk> .', -28.37084]
   [u'I love it , but it is not a <unk> , and it is not a <unk> .', -28.826918]

You can also try a larger beam size, such as 15.

.. code-block:: console

   $ python sequence_sampling.py --use-beam-search --bos I love it --beam_size 15 --print_num 15

Output is

.. code-block:: console

   Beam Seach Parameters: beam_size=15, alpha=0.0, K=5
   Generation Result:
   [u'I love it .', -1.1241297]
   [u'I love it " .', -4.001592]
   [u'I love it as a <unk> .', -8.038588]
   [u"I love it , and I don 't know how to do it .", -15.407309]
   [u"I love it , and I don 't want to do it .", -15.887625]
   [u"I love it , and I don 't know what it is .", -15.91673]
   [u"I love it , and I don 't know how to do so .", -16.780586]
   [u"I love it , and I don 't know how to do that .", -16.98329]
   [u"I love it , and I don 't think it is a <unk> .", -17.490877]
   [u"I love it , and I don 't think it would be a <unk> .", -19.416945]
   [u"I love it , and I don 't know how to do it , but I don 't know how to do it .", -28.04979]
   [u"I love it , and I don 't know how to do it , but I don 't think it is a <unk> .", -29.397102]
   [u"I love it , and I don 't know how to do it , but I don 't think it 's a good .", -29.406847]
   [u"I love it , and I don 't know how to do it , but I don 't think it is a good .", -29.413773]
   [u"I love it , and I don 't know how to do it , but I don 't think it 's a lot .", -29.43183]

Sequence Sampler
~~~~~~~~~~~~~~~~

Use the following command to decode to sample from the multinomial distribution, produced from softmax with temperature 1.0.

.. code-block:: console

   $ python sequence_sampling.py --use-sampling --bos I love it --beam_size 5 --print_num 5 --temperature 1.0

Output is

.. code-block:: console

   Sampling Parameters: beam_size=5, temperature=1.0
   Generation Result:
   [u'I love it and martial arts , history , and communism ; it is seems to be probably a date .', -76.772766]
   [u'I love it in all @-@ bodied households but like those who got part in the concept of refugee peoples , and had .', -96.42722]
   [u'I love it for adult people .', -17.899687]
   [u"I love it I think it 's through the side that we are going to mean the world it else .", -69.61122]
   [u'I love it in late arrangement .', -22.287495]

You can also try a lower temperature such as 0.95, which results in sharper distribution.

.. code-block:: console

   $ python sequence_sampling.py --use-sampling --bos I love it --beam_size 5 --print_num 5 --temperature 0.95

Output is

.. code-block:: console

   Sampling Parameters: beam_size=5, temperature=0.95
   Generation Result:
   [u'I love it .', -1.1241297]
   [u'I love it and then it pays me serious from what he writes .', -45.79579]
   [u"I love it as if this was from now <unk> , good as to the grounds of ' Hoyt ' where it had .", -91.47732]
   [u'I love it be an action .', -19.657116]
   [u'I love it and now leads to his best resulted in a shift between the two were announced in 2006 .', -71.7838]
