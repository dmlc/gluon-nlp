#!/bin/bash
pip install -v -e .
python -m spacy download en
python -m spacy download de
python -m nltk.downloader all
