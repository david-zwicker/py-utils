'''
Created on Feb 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains functions that can be used to handle files and directories
'''

from __future__ import division

import contextlib
import os



@contextlib.contextmanager
def change_directory(path):
    """
    A context manager which changes the directory to the given
    path, and then changes it back to its previous value on exit. Copied from
    http://code.activestate.com/recipes/576620-changedirectory-context-manager/
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)



def ensure_directory_exists(folder):
    """ creates a folder if it not already exists """
    try:
        os.makedirs(folder)
    except OSError:
        # assume that the directory already exists
        pass



def get_module_path(module):
    """ returns the path to the module """
    path = os.path.dirname(os.path.dirname(module.__file__))
    return os.path.abspath(path)



def replace_in_file(infile, outfile=None, *args, **kwargs):
    """ reads in a file, replaces the given data using python formatting and
    writes back the result to a file.
    
    `infile` is the file to be read
    `outfile` determines the output file to which the data is written. If it is
        omitted, teh input file will be overwritten instead
    """
    if outfile is None:
        outfile = infile
    
    content = open(infile, 'r').read()
    content = content.format(*args, **kwargs)
    open(outfile, 'w').write(content)    
