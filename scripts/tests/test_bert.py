import os
import subprocess
import time

import pytest

@pytest.mark.serial
@pytest.mark.remote_required
def test_transformer():
    process = subprocess.check_call(['python', './scripts/bert/create_pretraining_data.py',
                                     '--input_file', './scripts/bert/sample_text.txt',
                                     '--output_dir', 'test/bert/out', 
                                     '--vocab', 'book_corpus_wiki_en_uncased',
                                     # TODO remove --do_lower_case
                                     '--do_lower_case',
                                     '--max_seq_length', '128',
                                     '--max_predictions_per_seq', '20',
                                     '--dupe_factor', '5',
                                     '--masked_lm_prob', '0.15',
                                     '--short_seq_prob', '0.1',
                                     '--verbose'])
    time.sleep(5)
