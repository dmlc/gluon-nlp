#!/bin/bash
pip install -v -e .
python setup.py build_ext --force --inplace --define CYTHON_TRACE
python -m spacy download en
python -m nltk.downloader all
