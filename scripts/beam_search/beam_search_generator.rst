Beam Search Generator
---------------------

:download:`[Download] </scripts/beam_search.zip>`

This script can be used to generate sentences using beam search from a pretrained language model.

Use the following command to generate the sentences

.. code-block:: console

   $ python beam_search_generator.py --bos I love it --beam_size 5

Output is

.. code-block:: console

   Beam Seach Parameters: beam_size=5, alpha=0.0, K=5
   Generation Result:
   ['I love it , but it is not clear that it will be difficult to do it , but it is not a .', 243.20294]
   ['I love it , but it is not clear that it will be difficult to do it , so it is not a .', 242.4809]
   ['I love it , but it is not clear that it will be difficult to do so , but it is not a .', 242.45113]

You can also try a larger beam size.

.. code-block:: console

   $ python beam_search_generator.py --bos I love it --beam_size 10

Output is

.. code-block:: console

   Beam Seach Parameters: beam_size=10, alpha=0.0, K=5
   Generation Result:
   ['I love it , but it is not possible to do it , but it is not impossible to do it , but .', 246.26108]
   ['I love it , but it is not possible to do it , but it is not impossible to do it , and .', 245.80142]
   ["I love it , but it is not possible to do it , but I 'm not going to do it , but .", 245.55646]

Try beam size equals to 15


.. code-block:: console

   $ python beam_search_generator.py --bos I love it --beam_size 15

Output is

.. code-block:: console

   Beam Seach Parameters: beam_size=15, alpha=0.0, K=5
   Generation Result:
   ["I love it , and I don 't know how to do it , but I don ’ t think it would be .", 274.9892]
   ["I love it , and I don 't know how to do it , but I don ’ t think it will be .", 274.63895]
   ["I love it , and I don 't know how to do it , but I don ’ t want to do it .", 274.61063]
