# -*- coding: utf-8 -*-
#
# documentation build configuration file, created by
# sphinx-quickstart on Thu Jul 23 19:40:08 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
import sys
import os, subprocess
import shlex
import recommonmark
import sphinx_gallery
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '..'))

# -- General configuration ------------------------------------------------

# Version information.
import gluonnlp as nlp
version = nlp.__version__
release = nlp.__version__

# General information about the project.
project = u'gluonnlp'
author = u'%s developers' % project
copyright = u'2018, %s' % author
github_doc_root = 'http://gluon-nlp.mxnet.io/{}/'.format(str(version))

# add markdown parser
CommonMarkParser.github_doc_root = github_doc_root
source_parsers = {
    '.md': CommonMarkParser
}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
]

doctest_global_setup = '''
import gluonnlp
import mxnet as mx
from mxnet import gluon
import numpy as np
import doctest
doctest.ELLIPSIS_MARKER = '-etc-'
'''

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

nbsphinx_kernel_name = 'python3'
nbsphinx_allow_errors = True
nbsphinx_timeout = 1200
html_sourcelink_suffix = ''

html_context = {
    'display_github': True,
    'github_user': 'dmlc',
    'github_repo': 'gluon-nlp',
    'github_version': 'master',
    'conf_py_path': '/docs/',
    'last_updated': False,
    'commit': True
}

nbsphinx_prolog = """
{% set paths = env.docname.split('/') %}

.. only:: html

    :download:`[Download] <{{ "../%s.zip"|format(paths[1]) }}>`
"""

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst', '.ipynb', '.md']

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# generate autosummary even if no references
autosummary_generate = True

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/gluon_white.png'

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/gluon_s2.png'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'examples/*/*/**.rst', 'model_zoo/*/*/**.rst']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme is set by the make target
html_theme = os.environ.get('GLUONNLP_THEME', 'rtd')

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
# only import rtd theme and set it if want to build docs locally
if not on_rtd and html_theme == 'rtd':
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# -- Options for LaTeX output ---------------------------------------------
latex_elements = {
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, '%s.tex' % project, project,
   author, 'manual'),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'mxnet': ('https://mxnet.apache.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('http://matplotlib.org/', None),
    'nltk': ('http://www.nltk.org/', None),
}


from sphinx_gallery.sorting import ExplicitOrder

examples_dirs = []
gallery_dirs = []

subsection_order = ExplicitOrder([])

def setup(app):
    app.add_config_value('recommonmark_config', {
        'url_resolver': lambda url: github_doc_root + url,
        'auto_doc_ref': True
            }, True)
    app.add_transform(AutoStructify)
    app.add_javascript('google_analytics.js')
    app.add_javascript('copybutton.js')


sphinx_gallery_conf = {
    'backreferences_dir': 'gen_modules/backreferences',
    'doc_module': ('gluonnlp', 'mxnet', 'numpy'),
'reference_url': {
    'gluonnlp': None,
    'numpy': 'http://docs.scipy.org/doc/numpy-1.9.1'},
    'examples_dirs': examples_dirs,
    'gallery_dirs': gallery_dirs,
    'subsection_order': subsection_order,
    'find_mayavi_figures': False,
    'filename_pattern': '.py',
    'expected_failing_examples': []
}

# Napoleon settings
napoleon_use_ivar = True

# linkcheck settings
import multiprocessing
linkcheck_ignore = [r'http[s]://apache-mxnet.s3.*']
linkcheck_retries = 3
linkcheck_workers = multiprocessing.cpu_count() / 2
