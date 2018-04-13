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
  parallel 'Python 2': {
    node {
      ws('workspace/gluon-nlp-py2') {
        checkout scm
        sh """#!/bin/bash
        conda env update -f env/py2.yml
        source activate gluon_nlp_py2
        conda list
        python -m spacy download en
        python -m nltk.downloader all
        python setup.py install
        nosetests -v --nocapture --with-timer tests/unittest
        """
      }
    }
  },
  'Python 3': {
    node {
      ws('workspace/gluon-nlp-py3') {
        checkout scm
        sh """#!/bin/bash
        conda env update -f env/py3.yml
        source activate gluon_nlp_py3
        conda list
        python -m spacy download en
        python -m nltk.downloader all
        python setup.py install
        nosetests -v --nocapture --with-timer tests/unittest
        """
      }
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
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64
      make -C docs clean
      make -C docs html
      if [[ ${env.BRANCH_NAME} == PR-* ]]; then
          echo "Uploading doc to s3://gluon-nlp-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/"
          aws s3 sync --delete docs/_build/html/ s3://gluon-nlp-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/ --acl public-read
          echo "Uploaded doc to http://gluon-nlp-staging.s3-accelerate.dualstack.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html"
      else
          echo "Uploading doc to s3://gluon-nlp-staging/${env.BRANCH_NAME}/"
          aws s3 sync --delete docs/_build/html/ s3://gluon-nlp/${env.BRANCH_NAME}/ --acl public-read
          echo "Uploaded doc to http://gluon-nlp.mxnet.io/${env.BRANCH_NAME}/index.html"
      fi
      """
    }
  }
}
