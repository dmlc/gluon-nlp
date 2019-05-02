// -*- mode: groovy -*-

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
// This file contains the steps that will be used in the
// Jenkins pipelines

utils = load('ci/jenkins/utils.groovy')

def sanity_lint(workspace_name, conda_env_name, path) {
  return ['Lint': {
    node {
      ws("workspace/${workspace_name}") {
        utils.init_git()
        sh """
        set -ex
        source ci/prepare_clean_env.sh ${conda_env_name}
        make lintdir=${path} lint
        set +ex
        """
      }
    }
  }]
}

def test_unittest(workspace_name, conda_env_name,
                  test_path, cov_path,
                  mark,
                  threads, gpu, skip_report) {
  capture_flag = env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no'
  node_type = gpu?NODE_LINUX_GPU:NODE_LINUX_CPU
  return ["${conda_env_name}: ${test_path} -m '${mark}'": {
    node(node_type) {
      ws("workspace/${workspace_name}") {
        utils.init_git()
        sh """
        set -ex
        source ci/prepare_clean_env.sh ${conda_env_name}
        pytest -v ${capture_flag} -n ${threads} -m '${mark}' --durations=30 --cov ${cov_path} ${test_path}
        set +ex
        """
        if (!skip_report) utils.publish_test_coverage('GluonNLPCodeCov')
      }
    }
  }]
}

def test_doctest(workspace_name, conda_env_name,
                 test_path, cov_path, threads) {
  capture_flag = env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no'
  return ["${conda_env_name}: doctest ${test_path}'": {
    node(NODE_LINUX_CPU) {
      ws("workspace/${workspace_name}") {
        utils.init_git()
        sh """
        set -ex
        source ci/prepare_clean_env.sh ${conda_env_name}
        pytest -v ${capture_flag} -n ${threads} --durations=30 --cov ${cov_path} --doctest-modules ${test_path}
        set +ex
        """
        utils.publish_test_coverage('GluonNLPCodeCov')
      }
    }
  }]
}

def create_website(workspace_name, conda_env_name) {
  if (env.BRANCH_NAME.startsWith('PR-')){
    enforce_linkcheck = 'false'
    bucket = 'gluon-nlp-staging'
    path = env.BRANCH_NAME+'/'+env.BUILD_NUMBER
  } else {
    enforce_linkcheck = 'true'
    bucket = 'gluon-nlp'
    path = env.BRANCH_NAME
  }
  return ["${conda_env_name}: website'": {
    node(NODE_LINUX_GPU) {
      ws("workspace/${workspace_name}") {
        utils.init_git()
        sh """
        set -ex
        source ci/prepare_clean_env.sh ${conda_env_name}
        make docs
        if [[ ${enforce_linkcheck} == true ]]; then
            make -C docs linkcheck SPHINXOPTS=-W
        else
            set +ex
            make -C docs linkcheck
        fi;

        ci/upload_doc.sh ${bucket} ${path}
        set +ex
        """
      }
    }
  }]
}

def post_website_link() {
  if (env.BRANCH_NAME.startsWith("PR-")) {
    node {
      pullRequest.comment("Job ${env.BRANCH_NAME}/${env.BUILD_NUMBER} is complete. \nDocs are uploaded to http://gluon-nlp-staging.s3-accelerate.dualstack.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html")
    }
  }
}

return this
