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
    skip_stdout_lines = 12
    
    
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
        code += b";exit;"
        return self._run_command([], stdin=code, **kwargs)
    
    
    def run_script(self, filename, **kwargs):
        """ runs the matlab script `filename` and returns the output """
        code = b"run('" + filename.encode('utf-8') + b"')"
        return self.run_code(code, **kwargs)
    

    def extract_output_cells(self, output):
        """ parse the `output` to extract all output cells """
        cells = []
        
        for line in output.split(b'\n'):
            if line.startswith(b'>>      '):
                cells.append(line[8:])
            elif cells:
                cells[-1] += line
                    
        return cells
    