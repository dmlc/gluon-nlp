#!/bin/bash
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

set -e
set -u
set -o pipefail

cd ${WORKDIR}
git clone https://github.com/apache/tvm tvm --recursive
cd ${WORKDIR}/tvm
# checkout a hash-tag
git checkout bf862d4c4355eae4f18d89b3b6b98ed0a2c18e9c
git submodule update --init --recursive


mkdir -p build
cp cmake/config.cmake build
echo set\(USE_LLVM llvm-config-10\) >> build/config.cmake
echo set\(USE_CUDA ON\) >> build/config.cmake
echo set\(USE_CUDNN ON\) >> build/config.cmake
echo set\(USE_CUBLAS ON\) >> build/config.cmake
echo set\(USE_GRAPH_EXECUTOR ON\) >> build/config.cmake
echo set\(USE_BLAS openblas\) >> build/config.cmake

cd build
cmake -GNinja -DCUDA_CUBLAS_LIBRARY=/usr/lib/x86_64-linux-gnu/libcublas.so ..
ninja

# install python binding
cd ..
cd python
python3 -m pip install -U -e . --user
