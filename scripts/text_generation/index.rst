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
         --lm-model gpt2_345m # options are {gpt2_117m, gpt2_345m} \
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
    ['I love it in reference to the northwestern country. replay Liberties were raised from the late 1943 to June <eos>', -89.459656]
    ['I love it to them. Very account suggests that there is no basis as to whether the constellations are <eos>', -72.687996]
    ['I love it for quick attempts. It does not have any factors, and [the cause] has <eos>', -64.87619]
    ['I love it one in the English language, and say it was not for English the same standard than <eos>', -71.51008]
    ['I love it to take care of her; following many attempts to appease the Canadian military and making some <eos>', -75.5512]

You can also try a lower temperature such as 0.95, which results in sharper distribution.

.. code-block:: console

   $ python sequence_sampling.py random-sample --bos 'I love it' --beam-size 5 --print-num 5 --temperature 0.95

Output is

.. code-block:: console

    Sampling Parameters: beam_size=5, temperature=0.95, use_top_k=None
    Generation Result:
    ['I love it and flew by <unk> (a <unk> colleague Due to his delicate and non-serious attacks <eos>', -85.825195]
    ['I love it in a short anticipated 1927 hiatus. As a result, it was able to withstand changes <eos>', -71.8867]
    ['I love it for analysis. <eos>', -15.78739]
    ['I love it his own. The total of one hundred lives of all other documented <unk> in the Congo <eos>', -68.57835]
    ['I love it in his Why My Woman to Get Out of Graham Your Way. <eos>', -65.74211]

Finally, you can also try to constrain the sampling to sample only from the top-k tokens.

.. code-block:: console

   $ python sequence_sampling.py random-sample --bos 'I love it' --beam-size 5 --print-num 5 --temperature 0.95 --use-top-k 800

Output is

.. code-block:: console

    Sampling Parameters: beam_size=5, temperature=0.95, use_top_k=800
    Generation Result:
    ['I love it. It is the same as the Old Age. The best known of this is the <eos>', -30.544556]
    ['I love it and had a weak start by a group of only three-year-old fans. <eos>', -44.970097]
    ['I love it ". <eos>', -4.725212]
    ['I love it with the <unk>. <eos>', -7.236909]
    ['I love it and its working-based <unk> ". <eos>', -25.340023]

Beam Search Generator
~~~~~~~~~~~~~~~~~~~~~

Use the following command to decode using beam search.

.. code-block:: console

   $ python sequence_sampling.py beam-search --bos 'I love it' --beam-size 5 --print-num 5

Output is

.. code-block:: console

    Beam Seach Parameters: beam_size=5, alpha=0.0, K=5
    Generation Result:
    ['I love it. <eos>', -2.6606221]
    ['I love it. "<eos>', -4.072001]
    ['I love it, and the <unk> of the <unk>. <eos>', -14.573]
    ['I love it, and the <unk> of the <unk>. The <unk> of the <unk>, the <unk>, <eos>', -28.968985]
    ['I love it, and the <unk> of the <unk>. The <unk> of the <unk>, the <unk> and <eos>', -30.064144]

You can also try a larger beam size, such as 15.

.. code-block:: console

   $ python sequence_sampling.py beam-search --bos 'I love it' --beam-size 15 --print-num 15

Output is

.. code-block:: console

    Beam Seach Parameters: beam_size=15, alpha=0.0, K=5
    Generation Result:
    ['I love it. <eos>', -2.6606221]
    ['I love it. "<eos>', -4.072001]
    ['I love it ". <eos>', -5.222643]
    ['I love it, and the <unk> of the <unk>. <eos>', -14.573]
    ['I love it. It was the first time in the history of the history of the history of the <eos>', -21.041868]
    ['I love it. It was the first time in the history of the history of the country. <eos>', -21.262276]
    ['I love it. It was the first time in the history of the history of the United States. <eos>', -21.826159]
    ['I love it. It was the first time in the history of the history of the world. <eos>', -21.930265]
    ['I love it. It was the first time in the history of the history of the country. The <eos>', -21.94392]
    ['I love it. It was the first time in the history of the history of the city. <eos>', -22.00894]
    ['I love it. It was the first time in the history of the history of the country that the <eos>', -22.152416]
    ['I love it. It was the first time in the history of the history of the United States, <eos>', -22.170143]
    ['I love it. It was the first time in the history of the history of the country, and <eos>', -22.188667]
    ['I love it. It was the first time in the history of the history of the United States that <eos>', -22.254015]
    ['I love it. It was the first time in the history of the history of the state. <eos>', -22.398975]
