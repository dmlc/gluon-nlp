import os
import sys
import time
import notedown
import nbformat

assert len(sys.argv) == 2, 'usage: input.md'

# timeout for each notebook, in sec
timeout = 40 * 60

# the files will be ignored for execution
ignore_execution = []

input_path = sys.argv[1]

# Change working directory to directory of input file
input_dir, input_fn = os.path.split(input_path)
os.chdir(input_dir)

output_fn = '.'.join(input_fn.split('.')[:-1] + ['ipynb'])

reader = notedown.MarkdownReader()

# read
with open(input_fn, encoding='utf-8', mode='r') as f:
    notebook = reader.read(f)

if not any([i in input_fn for i in ignore_execution]):
    tic = time.time()
    notedown.run(notebook, timeout)
    print('=== Finished evaluation in %f sec'%(time.time()-tic))

# write
# need to add language info to for syntax highlight
notebook['metadata'].update({'language_info': {'name': 'python'}})

notebook_json = nbformat.writes(notebook)

with open(output_fn, encoding='utf-8', mode='w') as f:
    f.write(notebook_json)
