Git Workflow Howtos
-------------------

This page introduces several common strategies that you will possibly use when you contribute to the codes for this repository.


-  `How to submit pull request <#how-to-submit-pull-request>`__
-  `How to resolve conflict with master <#how-to-resolve-conflict-with-master>`__
-  `How to combine multiple commits into one <#how-to-combine-multiple-commits-into-one>`__
-  `What is the consequence of force push <#what-is-the-consequence-of-force-push>`__


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

   -  Select "dmlc/gluon-nlp" with "master" branch as the base
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
