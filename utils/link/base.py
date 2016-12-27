'''
Created on Aug 14, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

import logging
import subprocess

import six



class ExecutableBase(object):
    ''' Base class for external programs '''

    name = 'program_name'
    standards_args = []
    _detected_path = None  # automatically detected path to the program
    

    def __init__(self, executable_path=None):
        """ initializes the external program interface """
        if executable_path is None:
            self.executable_path = self.find_executable()
        else:
            self.executable_path = executable_path


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
    def found_executable(cls):
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
        
        
    def _run_command(self, command, stdin=None, skip_stdout_lines=None,
                     **kwargs):
        """ runs the script adding `command` to the command line and piping
        `stdin` to its stdin. The function returns the text written to stdout
        and stderr of the script.
        
        `command` can be either a string or a list of strings. Both give the
            additional command line arguments
        `stdin` is a string that will be send to the program's stdin
        `skip_stdout_lines` defines the number of lines that will be removed
            from stdout (usually because they are some kind of startup message)
        """
        # build the command to run the program
        cmd = [self.executable_path] + self.standards_args
        if isinstance(command, six.string_types):
            cmd.append(command)
        else:
            cmd.extend(command)
            
        logging.debug('Command to be executed: %s', cmd)

        if stdin is None:        
            # run program in a separate process and capture output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, **kwargs)
            stdout, stderr = process.communicate()
            
        else:
            # run program in a separate process, send stdin, and capture output
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, **kwargs)
            stdout, stderr = process.communicate(stdin)

        # process output if necessary
        if skip_stdout_lines is None:
            skip_stdout_lines = self.skip_stdout_lines
        if skip_stdout_lines > 0:
            stdout = stdout.split(b"\n", skip_stdout_lines + 1)[-1]
        
        logging.debug('stdout:\n%s', stdout)
        logging.debug('stderr:\n%s', stderr)
        
        return stdout, stderr
    
        