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
MD2IPYNB = $(ROOTDIR)/docs/md2ipynb.py

flake8:
	flake8 . --exclude conda --count --select=E901,E999,F821,F822,F823 --show-source --statistics

pylint:
	pylint --rcfile=$(ROOTDIR)/.pylintrc gluonnlp scripts/*/*.py

restruc:
	python setup.py check --restructuredtext --strict

lint:
	make flake8
	make pylint
	make restruc

docs: release
	make -C docs html SPHINXOPTS=-W

clean:
	git clean -f -d -x --exclude="$(ROOTDIR)/tests/externaldata/*" --exclude=conda
	make -C docs clean

compile_notebooks:
	for f in $(shell find docs/examples -type f -name '*.md' -print) ; do \
		DIR=`dirname $$f` ; \
		BASENAME=`basename $$f` ; \
		echo $$DIR $$BASENAME ; \
		cd $$DIR ; \
		python $(MD2IPYNB) $$BASENAME ; \
		cd - ; \
	done;

dist_scripts:
	find scripts/* -type d -prune | grep -v 'tests\|__pycache__' | xargs -t -n 1 -I{} zip -r {}.zip {}

dist_notebooks: compile_notebooks
	find docs/examples/* -type d -prune | grep -v 'tests\|__pycache__' | xargs -t -n 1 -I{} zip -r {}.zip {} -x "*.md"

test:
	py.test -v --capture=no --durations=0  tests/unittest scripts

release: dist_scripts dist_notebooks
	python setup.py sdist
