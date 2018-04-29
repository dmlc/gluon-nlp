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

ROOTDIR = $(CURDIR)

pylint:
	pylint --rcfile=$(ROOTDIR)/.pylintrc gluonnlp scripts/*/*.py

docs: release
	make -C docs html

clean:
	rm -rf gluonnlp.egg-info build dist | true
	rm -rf tests/data | true
	rm scripts/*.zip | true
	rm docs/examples/*.zip | true
	make -C docs clean

dist_scripts:
	find scripts/* -type d -prune | grep -v 'tests\|__pycache__' | xargs -n 1 -I{} zip -r {}.zip {}

dist_notebooks:
	find docs/examples/* -type d -prune | grep -v 'tests\|__pycache__' | xargs -n 1 -I{} zip -r {}.zip {}

test:
	py.test -v --capture=no --durations=0  tests/unittest scripts

release: dist_scripts dist_notebooks
	python setup.py sdist
