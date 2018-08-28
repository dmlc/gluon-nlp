def prepare_clean_env(env_name) {
  sh """#!/bin/bash
  printenv
  git clean -f -d -x --exclude='tests/externaldata/*' --exclude='tests/data/*' --exclude=conda
  conda env update --prune -f env/${env_name}.yml -p conda/${env_name}
  conda activate ./conda/${env_name}
  conda list
  make clean
  """
}

def install_dep(env_name) {
  sh """#!/bin/bash
  conda activate ./conda/${env_name}
  python setup.py install --force
  python -m spacy download en
  python -m nltk.downloader all
  """
}

def run_test(env_name, folders, nproc, serial, capture_opt) {
  sh script:"""#!/bin/bash
  conda activate ./conda/${env_name}
  printenv
  py.test -v ${capture_opt} -n ${nproc} -m "${serial}" --durations=50 --cov=./ ${folders}
  """
}

def report_cov(name, flag) {
  sh """#!/bin/bash
  bash ./codecov.sh -c -F ${flag} -n ${name}
  """
}

def upload_doc(bucket, path) {
  sh """#!/bin/bash
  echo "Uploading doc to s3://${bucket}/${path}/"
  aws s3 sync --delete docs/_build/html/ s3://${bucket}/${path}/ --acl public-read
  echo "Uploaded doc to http://${bucket}.s3-accelerate.dualstack.amazonaws.com/${path}/index.html"
  """
}

stage("Sanity Check") {
  node {
    ws('workspace/gluon-nlp-lint') {
      checkout scm
      prepare_clean_env('pylint')
      sh """#!/bin/bash
      conda activate ./conda/pylint
      printenv
      make lint
      """
    }
  }
}

stage("Unit Test") {
  parallel 'Python 2': {
    node {
      withCredentials([string(credentialsId: 'GluonNLPCodeCov', variable: 'CODECOV_TOKEN')]) {
        ws('workspace/gluon-nlp-py2') {
          checkout scm
          prepare_clean_env('py2')
          install_dep('py2')
          run_test('py2', 'tests/unittest', '4', 'not serial',
           env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no')
          run_test('py2', 'tests/unittest', '0', 'serial',
           env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no')
          report_cov(env.BRANCH_NAME+'-py2', 'unittests')
          run_test('py2', 'scripts', '4', 'not serial',
           env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no')
          run_test('py2', 'scripts', '0', 'serial',
           env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no')
          report_cov(env.BRANCH_NAME+'-py2', 'integration')
        }
      }
    }
  },
  'Python 3': {
    node {
      withCredentials([string(credentialsId: 'GluonNLPCodeCov', variable: 'CODECOV_TOKEN')]) {
        ws('workspace/gluon-nlp-py3') {
          checkout scm
          prepare_clean_env('py3')
          install_dep('py3')
          run_test('py3', 'tests/unittest', '4', 'not serial',
           env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no')
          run_test('py3', 'tests/unittest', '0', 'serial',
           env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no')
          report_cov(env.BRANCH_NAME+'-py3', 'unittests')
          run_test('py3', 'scripts', '4', 'not serial',
           env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no')
          run_test('py3', 'scripts', '0', 'serial',
           env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no')
          report_cov(env.BRANCH_NAME+'-py3', 'integration')
        }
      }
    }
  },
  'Documentation': {
    node {
      ws('workspace/gluon-nlp-docs') {
        checkout scm
        retry(3) {
          prepare_clean_env('doc')
          install_dep('doc')
          sh """#!/bin/bash
          conda activate ./conda/doc
          export LD_LIBRARY_PATH=/usr/local/cuda/lib64
          printenv
          make docs
          """
        }
        if (env.BRANCH_NAME.startsWith("PR-")) {
          upload_doc('gluon-nlp-staging', env.BRANCH_NAME+'/'+env.BUILD_NUMBER)
        } else {
          upload_doc('gluon-nlp', env.BRANCH_NAME)
        }
      }
    }
  }
}

stage("Deploy") {
  node {
    ws('workspace/gluon-nlp-docs') {
      checkout scm
      if (env.BRANCH_NAME.startsWith("PR-")) {
        pullRequest.comment("Job ${env.BRANCH_NAME}/${env.BUILD_NUMBER} is complete. \nDocs are uploaded to http://gluon-nlp-staging.s3-accelerate.dualstack.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html")
      }
    }
  }
}
