Contribute
==========

GluonNLP community welcomes contributions from anyone! Latest documentation can be found `here <http://gluon-nlp.mxnet.io/master/index.html>`__.

There are lots of opportunities for you to become our `contributors <https://github.com/dmlc/gluon-nlp/blob/master/contributor.rst>`__:

- Ask or answer questions on `GitHub issues <https://github.com/dmlc/gluon-nlp/issues>`__.
- Propose ideas, or review proposed design ideas on `GitHub issues <https://github.com/dmlc/gluon-nlp/issues>`__.
- Improve the `documentation <http://gluon-nlp.mxnet.io/master/index.html>`__.
- Contribute bug reports `GitHub issues <https://github.com/dmlc/gluon-nlp/issues>`__.
- Write new `scripts <https://github.com/dmlc/gluon-nlp/tree/master/scripts>`__ to reproduce
  state-of-the-art results.
- Write new `examples <https://github.com/dmlc/gluon-nlp/tree/master/docs/examples>`__.
- Write new `public datasets <https://github.com/dmlc/gluon-nlp/tree/master/gluonnlp/data>`__
  (license permitting).
- Most importantly, if you have an idea of how to contribute, then do it!

For a list of open starter tasks, check `good first issues <https://github.com/dmlc/gluon-nlp/labels/good%20first%20issue>`__.

How-to
++++++

- `Make changes <#make-changes>`__

- `Contribute scripts <#contribute-scripts>`__

- `Contribute examples <#contribute-examples>`__

- `Contribute new API <#contribute-new-api>`__

- `Git Workflow Howtos <#git-workflow-howtos>`__

   -  `How to submit pull request <#how-to-submit-pull-request>`__
   -  `How to resolve conflict with
      master <#how-to-resolve-conflict-with-master>`__
   -  `How to combine multiple commits into
      one <#how-to-combine-multiple-commits-into-one>`__
   -  `What is the consequence of force
      push <#what-is-the-consequence-of-force-push>`__


Make changes
------------

Our package uses continuous integration and code coverage tools for verifying pull requests. Before
submitting, contributor should perform the following checks:

- `Lint (code style) check <https://github.com/dmlc/gluon-nlp/blob/master/Jenkinsfile#L6-L11>`__.
- `Py2 <https://github.com/dmlc/gluon-nlp/blob/master/Jenkinsfile#L23-L31>`__ and `Py3 <https://github.com/dmlc/gluon-nlp/blob/master/Jenkinsfile#L42-L50>`__ tests.

Contribute Scripts
------------------

The `scripts <http://gluon-nlp.mxnet.io/master/scripts/index.html>`__ in GluonNLP are typically
for reproducing state-of-the-art (SOTA) results, or for a simple and interesting application.
They are intended for practitioners who are familiar with the libraries to tweak and hack. For SOTA
scripts, we usually request training scripts to be uploaded `here <https://github.com/dmlc/web-data/tree/master/gluonnlp/logs>`__, and then linked to in the example documentation.

See `existing examples <https://github.com/dmlc/gluon-nlp/tree/master/scripts>`__.

Contribute Examples
-------------------

Our `examples <http://gluon-nlp.mxnet.io/master/examples/index.html>`__ are intended for people who
are interested in NLP and want to get better familiarized on different parts in NLP. In order for
people to easily understand the content, the code needs to be clean and readable, accompanied by
good quality writing.

See `existing examples <https://github.com/dmlc/gluon-nlp/tree/master/docs/examples>`__.

Contribute new API
------------------

There are several different types of APIs, such as *model definition APIs, public dataset APIs, and
building block APIs*.

*Model definition APIs* facilitate the sharing of pre-trained models. If you'd like to contribute
models with pre-trained weights, you can `open an issue <https://github.com/dmlc/gluon-nlp/issues/new>`__
and ping committers first, we will help with things such as hosting the model weights while you propose the patch.

*Public dataset APIs* facilitate the sharing of public datasets. Like model definition APIs, if you'd like to contribute
new public datasets, you can `open an issue <https://github.com/dmlc/gluon-nlp/issues/new>`__ and ping committers and review
the dataset needs. If you're unsure, feel free to open an issue anyway.

Finally, our *data and model building block APIs* come from repeated patterns in examples. It has the highest quality bar
and should always starts from a good design. If you have an idea on proposing a new API, we
encourage you to `draft a design proposal first <https://github.com/dmlc/gluon-nlp/labels/enhancement>`__, so that the community can help iterate.
Once the design is finalized, everyone who are interested in making it happen can help by submitting
patches. For designs that require larger scopes, we can help set up GitHub project to make it easier
for others to join.

Contribute Docs
---------------

Documentation is no less important than code. Good documentation delivers the correct message clearly and concisely.
If you see any issue in the existing documentation, a patch to fix is most welcome! To locate the
code responsible for the doc, you may use "View page source" in the top right corner, or the
"[source]" links after each API. Also, `git grep` works nicely if there's unique string.

Git Workflow Howtos
-------------------

How to submit pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Before submit, please rebase your code on the most recent version of
   master, you can do it by

.. code:: bash

    git remote add upstream https://github.com/dmlc/gluon-nlp
    git fetch upstream
    git rebase upstream/master

-  If you have multiple small commits, it might be good to merge them
   together(use git rebase then squash) into more meaningful groups.
-  Send the pull request!

   -  Fix the problems reported by automatic checks
   -  If you are contributing a new module or new function, add a test.

How to resolve conflict with master
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  First rebase to most recent master

.. code:: bash

    # The first two steps can be skipped after you do it once.
    git remote add upstream https://github.com/dmlc/gluon-nlp
    git fetch upstream
    git rebase upstream/master

-  The git may show some conflicts it cannot merge, say
   ``conflicted.py``.

   -  Manually modify the file to resolve the conflict.
   -  After you resolved the conflict, mark it as resolved by

   .. code:: bash

       git add conflicted.py

-  Then you can continue rebase by

.. code:: bash

    git rebase --continue

-  Finally push to your fork, you may need to force push here.

.. code:: bash

    git push --force

How to combine multiple commits into one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes we want to combine multiple commits, especially when later
commits are only fixes to previous ones, to create a PR with set of
meaningful commits. You can do it by following steps. - Before doing so,
configure the default editor of git if you haven’t done so before.

.. code:: bash

    git config core.editor the-editor-you-like

-  Assume we want to merge last 3 commits, type the following commands

.. code:: bash

    git rebase -i HEAD~3

-  It will pop up an text editor. Set the first commit as ``pick``, and
   change later ones to ``squash``.
-  After you saved the file, it will pop up another text editor to ask
   you modify the combined commit message.
-  Push the changes to your fork, you need to force push.

.. code:: bash

    git push --force

Reset to the most recent master
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can always use git reset to reset your version to the most recent
master. Note that all your ***local changes will get lost***. So only do
it when you do not have local changes or when your pull request just get
merged.

.. code:: bash

    git reset --hard [hash tag of master]
    git push --force

What is the consequence of force push
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous two tips requires force push, this is because we altered
the path of the commits. It is fine to force push to your own fork, as
long as the commits changed are only yours.
