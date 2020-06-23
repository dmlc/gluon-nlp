import argparse
import os
import time

import nbformat
import notedown

parser = argparse.ArgumentParser(description='Convert md file to ipynb files.')
parser.add_argument('input', help='input.md', type=str)
parser.add_argument('-d', '--disable_compute',
                    help='Disable computing python scripts', action="store_true")
args = parser.parse_args()

# timeout for each notebook, in sec
timeout = 90 * 60

# the files will be ignored for execution
ignore_execution = []

# Change working directory to directory of input file
input_dir, input_fn = os.path.split(args.input)
if input_dir:
    os.chdir(input_dir)

output_fn = '.'.join(input_fn.split('.')[:-1] + ['ipynb'])

reader = notedown.MarkdownReader()

# read
with open(input_fn, encoding='utf-8', mode='r') as f:
    notebook = reader.read(f)

if not any([i in input_fn for i in ignore_execution]):
    tic = time.time()
    if not args.disable_compute:
        notedown.run(notebook, timeout)
    print('=== Finished evaluation in %f sec' % (time.time() - tic))

# write
# need to add language info to for syntax highlight
notebook['metadata'].update({'language_info': {'name': 'python'}})

notebook_json = nbformat.writes(notebook)

with open(output_fn, encoding='utf-8', mode='w') as f:
    f.write(notebook_json)
