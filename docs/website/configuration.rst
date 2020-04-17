Preview GluonNLP Website Locally
-----------------------------------------------------------------

The GluonNLP docs website is at `release branch <https://gluon-nlp.mxnet.io>`__, or `master branch <https://gluon-nlp.mxnet.io/master/index.html>`__. Its source code is at `gluon-nlp <https://github.com/dmlc/gluon-nlp>`__.

Currently the GluonNLP website is constructed from the source code via CI automatically. Here I will share:

- the structure of files used for the website, and
- how to make changes to the website and preview the website

Website Structure
~~~~~~~~~~~~~~~~~

Currently the docs part contain four sections: Model Zoo, Examples, API and Community. It should be noted that the model zoo is a link redirecting to the ``scripts`` folder in the parent folder. The other three folders are used exclusively by the docs website. Also, three different sections use ``rst``, ``py``, ``md`` files and their composition for compiling - they are inconsistent. So when you work on different sections of the docs website, you should  pay attention to handle the different sections with care.

The main structure, the index file of the entire website, is written in ``rst`` format. It calls the index file of each different section separately. Before compiling the website, you should be aware that:

- ``rst`` files are static files, they are directly displayed to the website with further styles;
- ``md`` files are script files, the python scripts in these files will be executed and then stored into ``ipynb`` files before converting ``ipynb`` files into website files.

Or more specifically, the files in the examples folder will be further executed and converted into intermediate files before writing to the final HTML files, while those in other folders don’t need further conversion or computation.

Environment Configuration Instruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, I will give a step by step instruction on how to compile this website from scratch.

1. Preview website without displaying python output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the command from https://github.com/dmlc/gluon-nlp/blob/master/docs/README.txt to install the necessary packages.

.. code:: bash

    pip install sphinx>=1.5.5 sphinx-gallery sphinx_rtd_theme matplotlib Image recommonmark

Then use the command below to build the website locally, all the ``python`` scripts are skipped and there is no output for ``python`` code blocks:

.. code:: bash

    make docs_local MD2IPYNB_OPTION=-d

You will get full HTML result for the website after successful execution.

2. Preview website with python output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To accomplish this task, we recommend you to use the instance ``g4dn.xlarge`` on Amazon EC2. For convenience, you can search *deep learning* in the filter bar to select the deep learning-enabled machines, where you will have no need of installing addition drivers.

After you have got the machine and logged to the machine, you will need to configure the packages using the command below:

.. code:: bash

    git clone https://github.com/dmlc/gluon-nlp
    cd gluon-nlp
    pip3 install --user -e '.[extras,dev]'

If necessary, you might still need to configure the packages like below:

Use ``python3`` command to get into the python execution screen, and then type the commands below to install the necessary packages inside python:

.. code:: python

    import nltk
    nltk.download('perluniprops')
    nltk.download('nonbreaking_prefixes')
    nltk.download('punkt')

By now, you should have installed all the necessary packages for the website. You can use the command below for previewing the website locally with all the python output:

.. code:: bash

    make docs_local

