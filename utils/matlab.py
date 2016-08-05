'''
Created on Aug 2, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>

Code for interacting with matlab
'''

from __future__ import division

import subprocess
import six


MATLAB_PATH = "/Applications/MATLAB_R2015b.app/bin/matlab"



def _run_matlab_commandline(command, skip_startup_lines=12, **kwargs):
    """ runs the matlab script `filename` and returns the output """
    # build the command to run matlab
    cmd = [MATLAB_PATH,
           "-nojvm", "-nodisplay", "-nosplash", "-nodesktop"]
    if isinstance(command, six.string_types):
        cmd.append(command)
    else:
        cmd.extend(command)
    
    # run matlab in a separate process and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=False, **kwargs)
    stdout, stderr = process.communicate()
    
    # process output if necessary
    if skip_startup_lines > 0:
        stdout = stdout.split("\n", skip_startup_lines + 1)[-1]
    
    return stdout, stderr



def run_matlab_code(code, skip_startup_lines=12, **kwargs):
    """ runs the matlab script `filename` and returns the output """
    return _run_matlab_commandline("-r \"%s;exit;\"" % code, skip_startup_lines,
                                   **kwargs)



def run_matlab_script(filename, skip_startup_lines=12, **kwargs):
    """ runs the matlab script `filename` and returns the output """
    return _run_matlab_commandline(["-r", filename], skip_startup_lines,
                                   **kwargs)
