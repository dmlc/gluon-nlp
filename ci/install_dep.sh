#!/bin/bash
pip install -v -e .
python -m spacy download en
python -m nltk.downloader all
