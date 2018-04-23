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
