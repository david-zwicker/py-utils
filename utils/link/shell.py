'''
Created on Aug 9, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import pipes
import subprocess as sp
import logging

import six


shellquote = pipes.quote



def shell_join(split_command):
    """ inverse of shlex.split """
    return ' '.join(shellquote(arg) for arg in split_command)



def run_command_over_ssh(host, command, user=None, ssh_args=None):
    """ run a command on a remote host using ssh

    `command` is the actual command as a string exactly as it would be typed
        into the command line after logging into the remote host
    `user` is an optional username used for logging into the host. If a
        password needs to be specified, set this to "USER:PASSWORD".
    `ssh_args` is a list of extra arguments that are passed to ssh

    Returns the stdout and stderr of the remote call.
    """
    # build the ssh command
    args = ['ssh']
    # add extra arguments that might be passed
    if ssh_args is not None:
        if isinstance(ssh_args, six.string_types):
            raise ValueError('`ssh_args` must be a list of arguments that '
                             'can be passed to subprocess.Popen.')
            args += ssh_args
    # add the host, optionally including a username
    if user:
        args.append(user + '@' + host)
    else:
        args.append(host)
    # append the actual command to run on the remote host
    args.append(command)

    # run the ssh command
    logging.getLogger(__name__).info('Run command `%s`', shell_join(args))
    proc = sp.Popen(args, shell=False, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout_data, stderr_data = proc.communicate()

    return stdout_data, stderr_data
