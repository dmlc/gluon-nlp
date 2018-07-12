stage("Sanity Check") {
  node {
    ws('workspace/gluon-nlp-lint') {
      checkout scm
      sh """#!/bin/bash
      git clean -f -d -x --exclude='tests/externaldata/*' --exclude=conda
      conda env update --prune -f env/pylint.yml -p conda/lint
      source activate ./conda/lint
      conda list
      make clean
      make pylint && python setup.py check --restructuredtext --strict
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
        git clean -f -d -x --exclude='tests/externaldata/*' --exclude=conda
        conda env update --prune -f env/py2.yml -p conda/py2
        source activate ./conda/py2
        conda list
        python -m spacy download en
        python -m nltk.downloader all
        make clean
        python setup.py install
        py.test -v --capture=no --durations=0 --cov=gluonnlp --cov=scripts tests/unittest scripts
        """
      }
    }
  },
  'Python 3': {
    node {
      withCredentials([string(credentialsId: 'GluonNLPCodeCov', variable: 'CODECOV_TOKEN')]) {
        ws('workspace/gluon-nlp-py3') {
          checkout scm
          sh """#!/bin/bash
          git clean -f -d -x --exclude='tests/externaldata/*' --exclude=conda
          conda env update --prune -f env/py3.yml -p conda/py3
          source activate ./conda/py3
          conda list
          python -m spacy download en
          python -m nltk.downloader all
          make clean
          python setup.py install
          py.test -v --capture=no --durations=0 --cov=gluonnlp --cov=scripts tests/unittest scripts
          EXIT_STATUS=\$?
          bash ./codecov.sh
          exit \$EXIT_STATUS
          """
        }
      }
    }
  }
}

stage("Deploy") {
  node {
    ws('workspace/gluon-nlp-docs') {
      checkout scm
      sh """#!/bin/bash
      printenv
      git clean -f -d -x --exclude='tests/externaldata/*' --exclude=conda
      conda env update --prune -f env/doc.yml -p conda/docs
      source activate ./conda/docs
      conda list
      python setup.py install
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64
      make clean
      make docs
      """

      if (env.BRANCH_NAME.startsWith("PR-")) {
        sh """#!/bin/bash
        echo "Uploading doc to s3://gluon-nlp-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/"
        aws s3 sync --delete docs/_build/html/ s3://gluon-nlp-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/ --acl public-read
        echo "Uploaded doc to http://gluon-nlp-staging.s3-accelerate.dualstack.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html" """
        pullRequest.comment("Job ${env.BRANCH_NAME}/${env.BUILD_NUMBER} is complete. \nDocs are uploaded to http://gluon-nlp-staging.s3-accelerate.dualstack.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html")
      } else {
        sh """#!/bin/bash
        echo "Uploading doc to s3://gluon-nlp/${env.BRANCH_NAME}/"
        aws s3 sync --delete docs/_build/html/ s3://gluon-nlp/${env.BRANCH_NAME}/ --acl public-read
        echo "Uploaded doc to http://gluon-nlp.mxnet.io/${env.BRANCH_NAME}/index.html" """
      }
    }
  }
}
