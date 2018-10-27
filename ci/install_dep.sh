#!/bin/bash
python setup.py install --force
python -m spacy download en
python -m nltk.downloader all
