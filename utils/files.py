'''
Created on Feb 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains functions that can be used to handle files and directories
'''

from __future__ import division

import contextlib
import glob
import itertools
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



def pattern_alternatives(pattern):
    """ parse a glob pattern and yields all possible pattern combinations """
    alternatives = []
    group = None
    str_cur, alt_cur = '', ''
    
    # parse the pattern by going though every character
    for c in pattern:
        if c == '{':
            # begin new group of alternatives
            if group is not None:
                raise RuntimeError('Nested brackets are not supported in glob '
                                   'patterns with alternatives')
                
            group = []
            alt_cur = ''
            
        elif c == '}':
            # end group of alternatives
            if group is None:
                raise RuntimeError('An opening bracket is missing')
                
            group.append(alt_cur)
            group = [str_cur + alt for alt in group]
            alternatives.append(group)
            group = None
            alt_cur, str_cur = '', ''
        
        elif group is None:
            # we're currently not in a group
            str_cur += c
        
        else:
            if c == ',':
                # start a new alternative within the group
                group.append(alt_cur)
                alt_cur = ''
            else:
                # add the character to the current alternative
                alt_cur += c
                
    if group is not None:
        raise RuntimeError('There was an unclosed group')
        
    if not alternatives:
        # there were no groups in the pattern
        yield str_cur
        
    else:
        alternatives[-1] = [alt + str_cur for alt in alternatives[-1]]
    
        # yield all possible combinations
        for groups in itertools.product(*alternatives):
            yield ''.join(groups)



def glob_alternatives(pattern):
    """ extends the support of the python glob library to also allow using
    pattern alternatives.
    
    Examples:
        "*.{png,jpg}" searches for all images
        "{a,z}*.{png,jpg}" searches for desktop images that start with either an
            `a` or a `z`
    """
    for sub_pattern in pattern_alternatives(pattern):
        for res in glob.iglob(sub_pattern):
            yield res
    
    