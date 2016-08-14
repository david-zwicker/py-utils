'''
Created on Aug 2, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>

Code for interacting with matlab
'''

from __future__ import division

import glob
import sys
import os

from .base import ExecutableBase



class Matlab(ExecutableBase):
    """ class that connects to Matlab """ 
    
    name = 'Matlab'
    standards_args = ["-nojvm", "-nodisplay", "-nosplash", "-nodesktop"]
    skip_init_lines = 12
    
    
    def find_program(self):
        """ tries to locate matlab. If successful, the function returns the
        command to run matlab """
        # look for matlab in Applications folder on mac     
        if sys.platform == 'darwin':
            pattern = "/Applications/MATLAB_R?????.app/bin/matlab"
            choices = glob.glob(pattern)
            if choices:
                # return last item from sorted results
                return sorted(choices)[-1]
        
        # otherwise, look in all the application paths
        paths = os.environ.get("PATH", "").split(os.pathsep)
        if 'MATLABROOT' in os.environ:
            paths.insert(0, os.environ['MATLABROOT'])
    
        for path in paths:
            candidate = os.path.realpath(os.path.join(path, 'matlab'))
            if os.path.isfile(candidate):
                return candidate
            elif os.path.isfile(candidate + '.exe'):
                return candidate + '.exe'
    
        raise RuntimeError('Could not find Matlab')

    
    def run_code(self, code, **kwargs):
        """ runs matlab code and returns the output """
        return self._run_command("-r \"%s;exit;\"" % code, **kwargs)
    
    
    def run_script(self, filename, **kwargs):
        """ runs the matlab script `filename` and returns the output """
        return self._run_command(["-r", filename], **kwargs)
