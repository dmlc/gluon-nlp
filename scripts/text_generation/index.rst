Text Generation
---------------

:download:`[Download] </model_zoo/text_generation.zip>`

Sampling a Language Model
+++++++++++++++++++++++++

This script can be used to generate sentences using beam search or a sequence sampler, to sample from a pre-trained language model such as GPT-2. For example:

.. code-block:: console

   $ python sequence_sampling.py random-sample \
         --bos 'Deep learning and natural language processing' \
         --beam-size 1 --print-num 1 \
         --lm-model gpt2_345m \
         --max-length 1024

Output is

.. code-block:: console

    Sampling Parameters: beam_size=1, temperature=1.0, use_top_k=None
    Generation Result:
    ['Deep learning and natural language processing brought application choice in healthcare and perception of sounds and heat to new heights, enriching our physical communities with medical devices and creating vibrant cultures. Anecdote is slowly diminishing but is hardly obsolete nor more appealing than experience.Despite those last words of wisdom, most headset makers even spook us with the complexity and poor code quality. the hard set a mere $150 and beginner creates center for getting started. Temp cheap:\nPosted by Fleegu at 12:02 PM<|endoftext|>', -461.15128]

Sequence Sampler
~~~~~~~~~~~~~~~~

Use the following command to decode to sample from the multinomial distribution.

.. code-block:: console

   $ python sequence_sampling.py random-sample --bos 'I love it' --beam-size 5 --print-num 5

Output is

.. code-block:: console

   Sampling Parameters: beam_size=5, temperature=1.0, use_top_k=None
   Generation Result:
   [u'I love it and martial arts , history , and communism ; it is seems to be probably a date .', -76.772766]
   [u'I love it in all @-@ bodied households but like those who got part in the concept of refugee peoples , and had .', -96.42722]
   [u'I love it for adult people .', -17.899687]
   [u"I love it I think it 's through the side that we are going to mean the world it else .", -69.61122]
   [u'I love it in late arrangement .', -22.287495]

You can also try a lower temperature such as 0.95, which results in sharper distribution.

.. code-block:: console

   $ python sequence_sampling.py random-sample --bos 'I love it' --beam-size 5 --print-num 5 --temperature 0.95

Output is

.. code-block:: console

   Sampling Parameters: beam_size=5, temperature=0.95, use_top_k=None
   Generation Result:
   [u'I love it .', -1.1241297]
   [u'I love it and then it pays me serious from what he writes .', -45.79579]
   [u"I love it as if this was from now <unk> , good as to the grounds of ' Hoyt ' where it had .", -91.47732]
   [u'I love it be an action .', -19.657116]
   [u'I love it and now leads to his best resulted in a shift between the two were announced in 2006 .', -71.7838]

Finally, you can also try to constrain the sampling to sample only from the top-k tokens.

.. code-block:: console

   $ python sequence_sampling.py random-sample --bos 'I love it' --beam-size 5 --print-num 5 --temperature 0.95 --use-top-k 30

Output is

.. code-block:: console

   Sampling Parameters: beam_size=5, temperature=0.95, use_top_k=3000
   Generation Result:
   ['I love it . A few three years later , however , it was believed that the few would be <eos>', -42.490887]
   ['I love it and I @,@ 360 people worked on it . It is not and all the thing he <eos>', -60.17195]
   ['I love it as well . <eos>', -8.63681]
   ['I love it with in the same role , and do not be as well as <unk> . The tradition <eos>', -47.913414]
   ['I love it and / or not it was written I would have actually been a " time @-@ old <eos>', -58.00537]

Beam Search Generator
~~~~~~~~~~~~~~~~~~~~~

Use the following command to decode using beam search.

.. code-block:: console

   $ python sequence_sampling.py beam-search --bos 'I love it' --beam-size 5 --print-num 5

Output is

.. code-block:: console

   Beam Search Parameters: beam_size=5, alpha=0.0, K=5
   Generation Result:
   [u'I love it .', -1.1241297]
   [u'I love it " .', -4.001592]
   [u'I love it , but it is not a <unk> .', -15.624882]
   [u'I love it , but it is not a <unk> , but it is not a <unk> .', -28.37084]
   [u'I love it , but it is not a <unk> , and it is not a <unk> .', -28.826918]

You can also try a larger beam size, such as 15.

.. code-block:: console

   $ python sequence_sampling.py beam-search --bos 'I love it' --beam-size 15 --print-num 15

Output is

.. code-block:: console

   Beam Search Parameters: beam_size=15, alpha=0.0, K=5
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
