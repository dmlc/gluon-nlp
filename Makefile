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

docs: docs_local
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

markdown:
	for f in $(shell find scripts -type f -name 'README.md' -print) ; do \
		CURDIR=$$(dirname $$f) ; \
		TARGET=$$CURDIR/index.rst ; \
		pandoc $$f --from markdown --to rst --wrap=none --column=999 -s -o $$TARGET; \
	done;

docs_local: markdown distribute
	make -C docs html SPHINXOPTS=-W

clean:
	git clean -ff -d -x --exclude="$(ROOTDIR)/tests/data/*" --exclude="$(ROOTDIR)/conda/" --exclude="$(ROOTDIR)/.idea/"

compile_notebooks:
	for f in $(shell find docs/examples -type f -name '*.md' -print) ; do \
		DIR=$$(dirname $$f) ; \
		BASENAME=$$(basename $$f) ; \
		TARGETNAME=$${BASENAME%.md}.ipynb ; \
		echo $$DIR $$BASENAME $$TARGETNAME; \
		cd $$DIR ; \
		if [ -f $$TARGETNAME ]; then \
			echo $$TARGETNAME exists. Skipping compilation of $$BASENAME in Makefile. ; \
		else \
			python3 $(MD2IPYNB) $(MD2IPYNB_OPTION) $$BASENAME ; \
		fi ; \
		cd - ; \
	done;

dist_scripts:
	cd scripts && \
	find * -type d -prune | grep -v 'tests\|__pycache__' | xargs -t -n 1 -I{} zip -r {}.zip {}

dist_notebooks:
	cd docs/examples && \
	find * -type d -prune | grep -v 'tests\|__pycache__' | xargs -t -n 1 -I{} zip -r {}.zip {} -x "*.md" -x "__pycache__" -x "*.pyc" -x "*.txt" -x "*.log" -x "*.params" -x "*.npz" -x "*.json"

distribute: dist_scripts dist_notebooks
	python3 setup.py sdist
