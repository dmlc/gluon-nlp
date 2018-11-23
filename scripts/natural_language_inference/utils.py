"""
utils.py

Part of NLI scripts of gluon-nlp.
Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>
"""

import logging
import gluonnlp as nlp

def logging_config(logpath=None,
                   level=logging.DEBUG,
                   console_level=logging.INFO,
                   no_console=False):
    """
    Config the logging.
    """
    logger = logging.getLogger('nli')
    # Remove all the current handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(filename)s:%(funcName)s: %(message)s')

    if logpath is not None:
        print('All Logs will be saved to {}'.format(logpath))
        logfile = logging.FileHandler(logpath, mode='w')
        logfile.setLevel(level)
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)

    if not no_console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logger.addHandler(logconsole)
