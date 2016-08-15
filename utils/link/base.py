'''
Created on Aug 14, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

import subprocess

import six



class ExecutableBase(object):
    ''' Base class for external programs '''

    name = 'program_name'
    standards_args = []
    

    def __init__(self, program_path=None):
        """ initializes the external program interface """
        if program_path is None:
            self.program_path = self.find_program()
        else:
            self.program_path = program_path


    def find_program(self):
        raise RuntimeError('Could not find %s' % self.name)
        
        
    def _run_command(self, command, show_cmd=False, stdin=None,
                     skip_stdout_lines=None, **kwargs):
        """ runs the script adding `command` to the command line and piping
        `stdin` to its stdin. The function returns the text written to stdout
        and stderr of the script.
        
        `command` can be either a string or a list of strings. Both give the
            additional command line arguments
        `show_cmd` is a flag that if enabled, outputs the command for debugging
        `stdin` is a string that will be send to the program's stdin
        `skip_stdout_lines` defines the number of lines that will be removed from
            the stdout (usually because they are some kind of startup message)
        """
        # build the command to run the program
        cmd = [self.program_path] + self.standards_args
        if isinstance(command, six.string_types):
            cmd.append(command)
        else:
            cmd.extend(command)
            
        if show_cmd:
            print('Command to be executed:')
            print(cmd)

        if stdin is None:        
            # run program in a separate process and capture output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       shell=False, **kwargs)
            stdout, stderr = process.communicate()
            
        else:
            # run program in a separate process, send stdin, and capture output
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       shell=False, **kwargs)
            stdout, stderr = process.communicate(stdin)

        # process output if necessary
        if skip_stdout_lines is None:
            skip_stdout_lines = self.skip_stdout_lines
        if skip_stdout_lines > 0:
            stdout = stdout.split("\n", skip_stdout_lines + 1)[-1]
        
        return stdout, stderr       
        