'''
Created on Aug 2, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>

Code for interacting with matlab
'''

from __future__ import division

import glob
import subprocess
import sys
import os

import six



MATLAB_CMD = None



def determine_matlab_command():
    """ tries to locate matlab. If successful, the function returns the
    command to run matlab """
    # look for matlab in Applications folder on mac     
    if sys.platform == 'darwin':
        pattern = "/Applications/MATLAB_R?????.app/bin/matlab"
        choices = list(sorted(glob.glob(pattern)))
        if choices:
            return choices[-1]
    
    # otherwise, look in all the application paths
    paths = os.environ.get("PATH")
    if 'MATLABROOT' in os.environ:
        paths.insert(0, os.environ['MATLABROOT'])

    if paths:
        for path in paths.split(os.pathsep):
            candidate = os.path.realpath(os.path.join(path, 'matlab'))
            if os.path.isfile(candidate):
                return candidate
            elif os.path.isfile(candidate + '.exe'):
                return candidate + '.exe'

    raise RuntimeError('Could not find matlab')
    


def _run_matlab_commandline(command, skip_startup_lines=12, **kwargs):
    """ runs the matlab script `filename` and returns the output """
    # make sure we find matlab
    global MATLAB_CMD
    if MATLAB_CMD is None:
        MATLAB_CMD = determine_matlab_command()
    
    # build the command to run matlab
    cmd = [MATLAB_CMD,
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
