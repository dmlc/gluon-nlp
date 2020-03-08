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

1. Converting md files into ipynb files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.1 Using existing ipynb from the website
:::::::::::::::::::::::::::::::::::::::::

If you don’t modify the examples section, what you need to do here is not converting the md files manually(which is time-consuming), but to get the ipynb files that are already ready for use. They can be found by clicking on the “DOWNLOAD THIS TUTORIAL” button on each page of the examples, such as `the word embedding tutorial <https://gluon-nlp.mxnet.io/master/examples/word_embedding/word_embedding.html>`__. After downloading these files, please put them in the same folder along with each md file. And then *delete* the md files. After these steps, you can skip the remaining of this section and go to the next section.

1.2 Generating new ipynb from md files
:::::::::::::::::::::::::::::::::::::::::

If you need to modify the examples, then you will need a server to do all the calculation because converting a ``md`` file into a ``ipynb`` file means running all the scripts in it and then save all the outputs.

To accomplish this task, we recommend you to use the instance ``g4dn.xlarge`` on Amazon EC2. For convenience, you can search *deep learning* in the filter bar to select the deep learning-enabled machines, where you will have no need of installing addition drivers.

After you have got the machine and logged to the machine, you will see that on the welcome screen, it shows the conda environments if you have selected a deep learning machine. Here we will choose the MXNet one, use the command below:

.. code:: bash

    source activate mxnet_p36

We will use the commands below to update the packages on this machine

.. code:: bash

    sudo apt-get update
    sudo apt-get upgrade

Use the commands below to install or update necessaryy packages:

.. code:: bash

    pip install notedown
    pip install --upgrade sphinx
    pip install --user -U numpy
    pip install "nltk==3.2.5"

After installing the packages above, get the GluonNLP repository from GitHub via the command:

.. code:: bash

    git clone https://github.com/dmlc/gluon-nlp

Then navigate to the gluon-nlp folder

.. code:: bash

    cd gluon-nlp

Use the command below to install the necessary packages in python:

.. code:: bash

    pip install --user -e '.[dev]'

If necessary, you might still need to configure the packages like below:

Use ``python`` command to get into the python execution screen, and then type the commands below to install the necessary packages inside python:

.. code:: bash

    import nltk
    nltk.download('perluniprops')
    nltk.download('nonbreaking_prefixes')
    nltk.download('punkt')

After all these configuration, you will be able to make the conversion from ``md`` files to the ``ipynb`` files.

Use the command like the one below to do the conversion:


.. code:: bash

    python3 docs/md2ipynb.py docs/examples/language_model/language_model.md

And then you will be able to see the result file in the same path as the ``md`` file.

2. Using make docs to convert static files into HTML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This work can be done on any machine as it doesn't need running the python scripts.

*Requirements*: You have the ``ipynb`` files ready. If you modified the ``md`` files, then you will need the first step to compile the new ``ipynb`` files.

It is also possible that you will need to install some necessary packages to help the ``make docs`` work. This include: python, pip and some other packages which will be installed through pip.

Use the command from https://github.com/dmlc/gluon-nlp/blob/master/docs/README.txt to install the necessary packages.

.. code:: bash

    pip install sphinx>=1.5.5 sphinx-gallery sphinx_rtd_theme matplotlib Image recommonmark

After successful installation and placing the ``ipynb`` files into correct places, plus deleting the ``md`` files, you should be able to successfully generate the output ``HTML`` files.
