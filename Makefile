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
	flake8 --exclude conda,*tests*,test_*.py,scripts/word_embeddings/tools/extern --count --select=E901,E999,F821,F822,F823 --show-source --statistics $(lintdir)

pylint:
	pylint --rcfile=$(ROOTDIR)/.pylintrc $(lintdir)

restruc:
	python setup.py check --restructuredtext --strict

lint:
	make lintdir=$(lintdir) flake8
	make lintdir=$(lintdir) pylint
	make restruc

docs: compile_notebooks distribute
	make -C docs html SPHINXOPTS=-W
	for f in $(shell find docs/examples -type f -name '*.md' -print) ; do \
		FILE=`echo $$f | sed 's/docs\///g'` ; \
		DIR=`dirname $$FILE` ; \
		BASENAME=`basename $$FILE` ; \
		HTML_BASENAME=`echo $$BASENAME | sed 's/md/html/'` ; \
		IPYNB_BASENAME=`echo $$BASENAME | sed 's/md/ipynb/'` ; \
		TARGET_HTML="docs/_build/html/$$DIR/$$HTML_BASENAME" ; \
		echo "processing" $$BASENAME ; \
		sed -i "s/$$IPYNB_BASENAME/$$BASENAME/g" $$TARGET_HTML; \
	done;
	for f in $(shell find docs/model_zoo -type f -name '*.rst' -print) ; do \
		DIR=`dirname $$f` ; \
		BASENAME=`basename $$f` ; \
		HTML_BASENAME=`echo $$BASENAME | sed 's/rst/html/'` ; \
		TARGET_HTML="docs/_build/html/$$DIR/$$HTML_BASENAME" ; \
		echo "processing" $$BASENAME ; \
		sed -i "s/docs\/model_zoo/scripts/g" $$TARGET_HTML; \
	done;
	sed -i.bak 's/33\,150\,243/23\,141\,201/g' docs/_build/html/_static/material-design-lite-1.3.0/material.blue-deep_orange.min.css;
	sed -i.bak 's/2196f3/178dc9/g' docs/_build/html/_static/sphinx_materialdesign_theme.css;

clean:
	git clean -ff -d -x --exclude="$(ROOTDIR)/tests/externaldata/*" --exclude="$(ROOTDIR)/tests/data/*" --exclude="$(ROOTDIR)/conda/"

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
	cd scripts && \
	find * -type d -prune | grep -v 'tests\|__pycache__' | xargs -t -n 1 -I{} zip -r {}.zip {}

dist_notebooks:
	cd docs/examples && \
	find * -type d -prune | grep -v 'tests\|__pycache__' | xargs -t -n 1 -I{} zip -r {}.zip {} -x "*.md" -x "__pycache__" -x "*.pyc" -x "*.txt" -x "*.log" -x "*.params" -x "*.npz" -x "*.json"

test:
	py.test -v --capture=no --durations=0  tests/unittest scripts

distribute: dist_scripts dist_notebooks
	python setup.py sdist
