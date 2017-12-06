'''
Created on Aug 14, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

import logging
import os
import subprocess

import six



class ExecutableBase(object):
    ''' Base class for external programs '''

    name = 'program_name'
    standards_args = []
    skip_stdout_lines = 0    
    _detected_path = None  # automatically detected path to the program


    def __init__(self, executable_path=None):
        """ initializes the external program interface """
        if executable_path is None:
            self.executable_path = self.find_executable()
        else:
            self.executable_path = executable_path
            
        self.stdout = None
        self.stderr = None
            
        self._logger = logging.getLogger(self.__class__.__module__)


    @classmethod
    def _find_executable(cls):
        """ find the path to the program automatically. This method should be
        overwritten to implement custom search algorithms """
        raise NotImplementedError('Do not know how to find program `%s`' %
                                  cls.name)
        
        
    @classmethod
    def find_executable(cls):
        """ detect program path automatically """
        if cls._detected_path is None:
            # detect the path
            cls._detected_path = cls._find_executable()
            
        return cls._detected_path
    
    
    @classmethod
    def is_available(cls):
        """ returns True if the program could be found. This swallows all
        exceptions that are raised during the automatic detection of the
        executable and may thus mask problems during the detection. Instead, use
        `find_executable` to discover the path and any problems."""
        try:
            cls.find_executable()
        except:
            return False
        else:
            return True  
        
        
    def log_output(self, level=logging.DEBUG):
        """ logs the output of a command """
        self._logger.log(level, 'STDOUT:\n%s', self.stdout)
        self._logger.log(level, 'STDERR:\n%s', self.stderr)
                
        
    def _run_command(self, command, stdin=None, skip_stdout_lines=None,
                     environment=None, shell=False, **kwargs):
        """ runs the script adding `command` to the command line and piping
        `stdin` to its stdin. The function returns the text written to stdout
        and stderr of the script.
        
        `command` can be either a string or a list of strings. Both give the
            additional command line arguments
        `stdin` is a string that will be send to the program's stdin
        `skip_stdout_lines` defines the number of lines that will be removed
            from stdout (usually because they are some kind of startup message)
        `environment` is a dictionary of environment variables that are set on
            top of the current environment
        `shell` determines whether a separate shell is invoked
            
        All additional keyword arguments are forwarded to the call of
        `subprocess.Popen`.
        """
        # build the command to run the program
        cmd = [self.executable_path] + self.standards_args
        if isinstance(command, six.string_types):
            cmd.append(command)
        else:
            cmd.extend(command)
        self._logger.debug('Command to be executed: %s', cmd)

        # get environment to run the process
        if environment is not None:
            process_env = os.environ.copy()
            process_env.update(environment)
        else:
            process_env = None
            
        if shell and not isinstance(cmd, six.string_types):
            # Using a shell requires us to convert the command list to a string
            cmd = ' '.join(cmd)
            
        if stdin is None:        
            # run program in a separate process and capture output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       env=process_env, shell=shell,
                                       **kwargs)
            self.stdout, self.stderr = process.communicate()
            
        else:
            # run program in a separate process, send stdin, and capture output
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       env=process_env, shell=shell,
                                       **kwargs)
            self.stdout, self.stderr = process.communicate(stdin)

        # process output if necessary
        if skip_stdout_lines is None:
            skip_stdout_lines = self.skip_stdout_lines
        if skip_stdout_lines > 0:
            self.stdout = self.stdout.split(b"\n", skip_stdout_lines + 1)[-1]

        self.log_output(level=logging.DEBUG)
        
        return self.stdout, self.stderr
    
        