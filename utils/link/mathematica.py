'''
Created on Aug 14, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>

Code for interacting with Mathematica
'''

from __future__ import division

import sys
import os

from .base import ExecutableBase



class Mathematica(ExecutableBase):
    """ class that connects to Mathematica """ 
    
    name = 'Mathematica'
    standards_args = ["-noprompt"]
    
    
    def find_program(self):
        """ tries to locate Mathematica. If successful, the function returns the
        command to run Mathematica
        
        This function is looking for `MathKernel` on MacOS, `math` on linux, and
        `MathKernel.exe` on windows
        """
        if sys.platform == 'darwin':
            # look for Mathematica in Applications folder on mac     
            candidate = "/Applications/Mathematica.app/Contents/MacOS/MathKernel"
            if os.path.isfile(candidate):
                return candidate
            
        elif sys.platform == 'win32':
            # search `MathKernel.exe` in all paths
            paths = os.environ.get("PATH", "").split(os.pathsep)
            for path in paths:
                candidate = os.path.realpath(os.path.join(path, 'MathKernel.exe'))
                if os.path.isfile(candidate):
                    return candidate
                
        else:
            # search `math` in all paths
            paths = os.environ.get("PATH", "").split(os.pathsep)
            for path in paths:
                candidate = os.path.realpath(os.path.join(path, 'math'))
                if os.path.isfile(candidate):
                    return candidate
        
        raise RuntimeError('Could not find Mathematica')
    

    def run_code(self, code, **kwargs):
        """ runs Mathematica code and returns the output """
        return self._run_command(["-run", "%s" % code], **kwargs)
        

    def run_script(self, filename, **kwargs):
        """ runs the Mathematica script `filename` and returns the output """
        return self._run_command(["-run", "<<%s" % filename], **kwargs)
