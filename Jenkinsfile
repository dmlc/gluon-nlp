stage("Sanity Check") {
  node {
    ws('workspace/gluon-nlp-lint') {
      checkout scm
      sh """#!/bin/bash
      conda env update -f env/pylint.yml
      source activate gluon_nlp_pylint
      conda list
      make pylint
      """
    }
  }
}

stage("Unit Test") {
  node {
    ws('workspace/gluon-nlp-py2') {
      checkout scm
      sh """#!/bin/bash
      conda env update -f env/py2.yml
      source activate gluon_nlp_py2
      conda list
      python -m spacy download en
      python setup.py install
      nosetests -v --nocapture --with-timer tests/unittest
      """
    }
  }
  node {
    ws('workspace/gluon-nlp-py3') {
      checkout scm
      sh """#!/bin/bash
      conda env update -f env/py3.yml
      source activate gluon_nlp_py3
      conda list
      python -m spacy download en
      python setup.py install
      nosetests -v --nocapture --with-timer tests/unittest
      """
    }
  }
}

stage("Deploy") {
  node {
    ws('workspace/gluon-nlp-docs') {
      checkout scm
      sh """#!/bin/bash
      conda env update -f env/doc.yml
      source activate gluon_nlp_docs
      conda list
      python setup.py install
      make docs
      if [[ ${env.BRANCH_NAME} == PR-* ]]; then
          echo "Uploading doc to s3://gluon-nlp-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/"
          aws s3 sync --delete docs/_build/html/ s3://gluon-nlp-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/ --acl public-read
      else
          echo "Uploading doc to s3://gluon-nlp-staging/${env.BRANCH_NAME}/"
          aws s3 sync --delete docs/_build/html/ s3://gluon-nlp/${env.BRANCH_NAME}/ --acl public-read
      fi
      """
    }
  }
}
