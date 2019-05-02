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

def test_unittest(workspace_name, conda_env_name, test_path, cov_path, mark, threads, gpu) {
  capture_flag = env.BRANCH_NAME.startsWith('PR-')?'':'--capture=no'
  cov_flag = env.BRANCH_NAME.startsWith('PR-')?('PR'+env.CHANGE_ID):env.BRANCH_NAME
  node_type = gpu?NODE_LINUX_GPU:NODE_LINUX_CPU
  return ["Python2: ${test_path} -m '${mark}'": {
    node(node_type) {
      ws("workspace/${workspace_name}") {
        utils.init_git()
        sh """
            set -ex
            source ci/prepare_clean_env.sh ${conda_env_name}
            pytest -v ${capture_flag} -n ${threads} -m '${mark}' --durations=30 ${test_path} --cov ${cov_path}
            set +ex
        """
        utils.publish_test_coverage('GluonNLPCodeCov')
      }
    }
  }]
}

return this
