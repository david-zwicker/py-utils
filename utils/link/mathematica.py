'''
Created on Aug 14, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>

Code for interacting with Mathematica
'''

from __future__ import division

import collections
import sys
import os

from .base import ExecutableBase



class Mathematica(ExecutableBase):
    """ class that connects to Mathematica """ 
    
    name = 'Mathematica'
    standards_args = []
    skip_stdout_lines = 0
    
    
    @classmethod
    def _find_program(cls):
        """ tries to locate Mathematica. If successful, the function returns the
        command to run Mathematica
        
        This function is looking for `MathKernel` on MacOS, `math` on linux, and
        `MathKernel.exe` on windows
        """
        if sys.platform == 'darwin':
            # look for Mathematica in Applications folder on mac     
            candidate = \
                    "/Applications/Mathematica.app/Contents/MacOS/MathKernel"
            if os.path.isfile(candidate):
                return candidate
            
        elif sys.platform == 'win32':
            # search `MathKernel.exe` in all paths
            paths = os.environ.get("PATH", "").split(os.pathsep)
            for path in paths:
                candidate = os.path.join(path, 'MathKernel.exe')
                candidate = os.path.realpath(candidate)
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
        kwargs.setdefault('skip_stdout_lines', 2)
        code += b'\nExit[];'  # make sure the program exits
        return self._run_command("", stdin=code, **kwargs)
        

    def run_script(self, filename, **kwargs):
        """ runs the Mathematica script `filename` and returns the output """
        return self._run_command(["-noprompt", "-run", "<<%s" % filename],
                                 **kwargs)
    
    
    def extract_output_cell(self, output, cell_nr):
        """ parse the `output` to extract the output cell given by `cell_nr` """
        # define the token the signifies the begin of the output cell
        token = ('Out[%d]= ' % cell_nr).encode('utf-8')
    
        res = None
        for line in output.split(b'\n'):
            if line.startswith(token):
                # found the beginning of the output cell
                res = line[len(token):]
            elif res is not None:
                # the output cell has been found
                if line.startswith(b'In') or line.startswith(b'Out'):
                    # a new cell begins and we're thus finished
                    break
                else:
                    # append the output, since it belongs to the target cell
                    res += b'\n' + line
        
        return res.rstrip()
    
    
    def extract_output_cells(self, output):
        """ parse the `output` to extract all output cells """
        # collect all cells in an ordered dictionary
        cells = collections.OrderedDict()
        
        cell_id, cell_content = None, None
        for line in output.split(b'\n'):
            if line.startswith(b'Out['):
                # store the old cell
                if cell_id is not None:
                    cells[cell_id] = cell_content.rstrip()
                # extract cell index
                cell_header = line.split(b']')[0]
                cell_id = int(cell_header[4:])
                cell_content = line[len(cell_header) + 3:]
    
            elif cell_id is not None:
                # the output cell has been found
                if line.startswith(b'In['):
                    # store the old cell
                    if cell_id is not None:
                        cells[cell_id] = cell_content.rstrip()
                    # we don't store the input cells
                    cell_id, cell_content = None, None
                else:
                    # append the output, since it belongs to the cell
                    cell_content += b'\n' + line
    
        # store the last cell
        if cell_id is not None:
            cells[cell_id] = cell_content.rstrip()
                    
        return cells
        