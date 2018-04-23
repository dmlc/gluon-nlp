Beam Search Generator
---------------------

This script can be used to generate sentences using beam search from a pretrained language model.

Use the following command to generate the sentences

.. code-block:: bash

   $ python beam_search_generator.py --bos I love it --beam_size 5

Output is

.. code-block:: log

    ['I love it , but it is not clear that it will be difficult to do it , but it is not .', 243.01672]
    ['I love it , but it is not clear that it will be difficult to do so , but it is not .', 242.37096]
    ['I love it , but it is not clear that it will be difficult to do it , so it is not .', 242.17531]
    ['I love it , but it is not clear that it will be difficult to do it , but it is a .', 241.43657]
    ["I love it , but it is not clear that it will be difficult to do it , but it 's not .", 241.37198]

You can also try a larger beam size.

.. code-block:: bash

   $ python beam_search_generator.py --bos I love it --beam_size 10

Output is

.. code-block:: log

    ['I love it , but it is not possible to do it , but it is not impossible to do it , but .', 246.26108]
    ['I love it , but it is not possible to do it , but it is not impossible to do it , and .', 245.80142]
    ["I love it , but it is not possible to do it , but I 'm not going to do it , but .", 245.55646]
    ['I love it , but it is not possible to do it , but it is not impossible to do so , and .', 245.44412]
    ['I love it , but it is not possible to do it , but it is not impossible to do so , but .', 245.37302]
    ["I love it , but it is not possible to do it , but I 'm not going to do it , and .", 245.2199]
    ['I love it , but it is not possible to do it , but it is not impossible to do it , so .', 244.95819]
    ['I love it , but it is not possible to do it , but it is not impossible to do it , because .', 244.92368]
    ['I love it , but it is not possible to do it , but it is not impossible to do it , or .', 244.8313]
    ["I love it , but it is not possible to do it , but I 'm not going to do anything for the .", 244.75426]

Try beam size equals to 15


.. code-block:: bash

   $ python beam_search_generator.py --bos I love it --beam_size 15

Output is

.. code-block:: log

    ["I love it , and I don 't know how to do it , but I don ’ t think it would be .", 274.9892]
    ["I love it , and I don 't know how to do it , but I don ’ t think it will be .", 274.63895]
    ["I love it , and I don 't know how to do it , but I don ’ t want to do it .", 274.61063]
    ["I love it , and I don 't know how to do it , but I don ’ t want to do anything .", 274.32306]
    ["I love it , and I don 't know how to do it , but I don 't know how to do it .", 273.984]
    ["I love it , and I don 't know how to do it , but I don ’ t think it 's a .", 273.90735]
    ["I love it , and I don 't know how to do it , but I don ’ t think it is a .", 273.80078]
    ["I love it , and I don 't know how to do it , but I don 't think it would be a .", 273.35507]
    ["I love it , and I don 't know how to do it , but I don ’ t want to do a .", 273.24527]
    ["I love it , and I don 't know how to do it , but I don ’ t want to do any .", 273.2016]
    ["I love it , and I don 't know how to do it , but I don ’ t know that it is .", 273.0937]
    ["I love it , and I don 't know how to do it , but I don ’ t think it was a .", 272.85138]
    ["I love it , and I don 't know how to do it , but I don ’ t want to do so .", 272.84656]
    ["I love it , and I don 't know how to do it , but I don ’ t think it would have .", 272.81784]
    ["I love it , and I don 't know how to do it , but I don ’ t want to do this .", 272.7752]