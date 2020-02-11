# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Test inference with BERT checkpoints"""
import pytest
import zipfile
import subprocess
import sys
import re
import mxnet as mx

@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.remote_required
@pytest.mark.integration
def test_bert_checkpoints():
    script = './scripts/bert/finetune_classifier.py'
    param = 'bert_base_uncased_sst-a628b1d4.params'
    param_zip = 'bert_base_uncased_sst-a628b1d4.zip'
    arguments = ['--log_interval', '1000000', '--model_parameters', param,
                 '--gpu', '0', '--only_inference', '--task_name', 'SST',
                 '--epochs', '1']
    url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/models/' + param_zip
    mx.gluon.utils.download(url , path='.')
    with zipfile.ZipFile(param_zip) as zf:
        zf.extractall('.')
    p = subprocess.check_call([sys.executable, script] + arguments)
    with open('log_SST.txt', 'r') as f:
        x = f.read()
        find = re.compile('accuracy:0.[0-9]+').search(str(x)).group(0)
        assert float(find[len('accuracy:'):]) > 0.92
