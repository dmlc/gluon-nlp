stage("Sanity Check") {
  node {
    ws('workspace/gluon-nlp-lint') {
      checkout scm
      sh('ci/step_sanity_check.sh')
    }
  }
}

stage("Unit Test") {
  capture_flag = env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no'
  cov_flag = env.BRANCH_NAME.startsWith('PR-')?('PR'+env.CHANGE_ID):env.BRANCH_NAME
  parallel 'Python 2': {
    node {
      withCredentials([string(credentialsId: 'GluonNLPCodeCov', variable: 'CODECOV_TOKEN')]) {
        ws('workspace/gluon-nlp-py2') {
          checkout scm
          sh("ci/step_unit_test.sh py2 ${cov_flag} ${capture_flag}")
        }
      }
    }
  },
  'Python 3': {
    node {
      withCredentials([string(credentialsId: 'GluonNLPCodeCov', variable: 'CODECOV_TOKEN')]) {
        ws('workspace/gluon-nlp-py3') {
          checkout scm
          sh("ci/step_unit_test.sh py3 ${cov_flag} ${capture_flag}")
        }
      }
    }
  },
  'Python 3 Master': {
    node {
      withCredentials([string(credentialsId: 'GluonNLPCodeCov', variable: 'CODECOV_TOKEN')]) {
        ws('workspace/gluon-nlp-py3-master') {
          checkout scm
          sh("ci/step_unit_test.sh py3-master ${cov_flag} ${capture_flag}")
        }
      }
    }
  },
  'DocTest': {
    node {
      ws('workspace/gluon-nlp-doctest') {
        checkout scm
        sh("ci/step_doc_test.sh")
      }
    }
  },
  'Documentation': {
    node {
      ws('workspace/gluon-nlp-docs') {
        checkout scm
        if (env.BRANCH_NAME.startsWith("PR-")) {
          sh("ci/step_documentation.sh false")
          bucket = 'gluon-nlp-staging'
          path = env.BRANCH_NAME+'/'+env.BUILD_NUMBER
        } else {
          sh("ci/step_documentation.sh true")
          bucket = 'gluon-nlp'
          path = env.BRANCH_NAME
        }
        sh("ci/upload_doc.sh ${bucket} ${path}")
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
