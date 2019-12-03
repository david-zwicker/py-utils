'''
Created on Aug 9, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import pipes
import subprocess as sp
import logging



shellquote = pipes.quote



def shell_join(split_command):
    """ inverse of shlex.split """
    return ' '.join(shellquote(arg) for arg in split_command)



def run_command_over_ssh(host, command, user=None):
    """ run a command on a remote host, optionally using a username
    
    If a password needs to be specified, use `user="USER:PASSWORD"`. Note that
    this is an unsafe function since it uses `shell=True` in the subprocess
    call. Never use this function with untrusted input!
    
    Returns the stdout and stderr of the remote call.
    """
    # build the ssh command
    args = ['ssh',  # run command
            '-t']   # allocate pseudo-terminal, to use e.g. cd command
    if user:
        args.append(user + '@' + host)
    else:
        args.append(host)
    args.append(command)
    
    # run the command 
    logging.getLogger(__name__).info('Run command `%s`', shell_join(args))
    proc = sp.Popen(args, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout_data, stderr_data = proc.communicate()
    
    return stdout_data, stderr_data
