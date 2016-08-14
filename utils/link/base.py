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
        
        
    def _run_command(self, command, show_cmd=False, **kwargs):
        """ runs the matlab script `filename` and returns the output """
        # make sure we find matlab
        # build the command to run matlab
        cmd = [self.program_path] + self.standards_args
        if isinstance(command, six.string_types):
            cmd.append(command)
        else:
            cmd.extend(command)
            
        if show_cmd:
            print('Command to be executed:')
            print(cmd)
        
        # run program in a separate process and capture output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=False,
                                   **kwargs)
        stdout, stderr = process.communicate()

        # process output if necessary
        if self.skip_init_lines > 0:
            stdout = stdout.split("\n", self.skip_init_lines + 1)[-1]
        
        return stdout, stderr       
        