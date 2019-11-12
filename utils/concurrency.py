'''
Created on Feb 15, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import select
import subprocess as sp
import os
import threading

import six



class WorkerThread(object):
    """ class that launches a worker thread as a daemon that applies a given
    function """
    
    def __init__(self, function, use_threads=True, synchronous=True):
        """ initializes the worker thread with the supplied function that will
        be called subsequently.
        `synchronous` is a flag determining whether the result from the worker
            thread will be synchronized with the input. If it is not, it can be
            that a call to `get` returns the result from a previous worker
            thread. Additionally, subsequent calls to `get` can return different
            results, even if no new calculation was initiated via `put`. 
        """
        self.function = function
        self.use_threads = use_threads
        self.synchronous = synchronous
        self._result = None
        
        if self.use_threads:
            self._event_start = threading.Event()
            self._event_finish = threading.Event()
            self._event_finish.set()
            self._thread = threading.Thread(target=self._worker_function,
                                            args=[self._event_start,
                                                  self._event_finish])
            self._thread.daemon = True
            self._thread.start()
        
        
    def _worker_function(self, event_start, event_finish):
        """ event loop of the worker thread """ 
        while True:
            # wait until starting signal is set
            event_start.wait()
            event_start.clear()
            # start the calculation
            self._result = self.function(*self._args, **self._kwargs)
            # signal that the calculation finished
            event_finish.set()
        
        
    def put(self, *args, **kwargs):
        """ starts the worker thread by passing the supplied arguments to it.
        Note that the arguments are not copied and should therefore not be
        modified while the thread is running in the background
        """
        # set the arguments for the call
        self._args = args
        self._kwargs = kwargs
        
        if self.use_threads:
            # wait until the worker is finished (in case it is still running)
            self._event_finish.wait()
            self._event_finish.clear()
            if self.synchronous:
                # reset the result variable if the result should be synchronized
                self._result = None
            # signal that the worker may begin
            self._event_start.set()
        
        else:
            # don't use threads and thus clear the result
            self._result = None
        
        
    def get(self):
        """ retrieves the result from the last job that was put """
        if self.use_threads:
            if self.synchronous or self._result is None:
                # wait until the worker finished
                self._event_finish.wait()
                # => the result is in self._result
                
        elif self._result is None:
            # calculate the result
            self._result = self.function(*self._args, **self._kwargs)
            
        # retrieve the result
        return self._result
    
    

class MonitorProcessOutput(object):
    """ class that starts a process and monitors its output while it is running
    
    Inspired by https://gist.github.com/mckaydavis/e96c1637d02bcf8a78e7
    """
    
    
    def __init__(self, args, env=None, timeout=0.1, bufsize=1024, stdin=False):
        """
        `args` is a list or string setting the program to call
        `env` can be a dictionary that defines additional environmental
            variables
        `timeout` determines the frequency of polling the results. A value of 
            `None` implies indefinite waiting time.
        `stdin` is a flag determining whether a pipe is opened to the stdin of
            the process. Alternatively, `stdin` can be a string, which is send
            to the process and stdin is closed afterwards.
        """
        self.timeout = timeout
        self.bufsize = bufsize
        
        # create pipes to receive stdout and stderr from process
        (self._pipe_out_r, self._pipe_out_w) = os.pipe()
        (self._pipe_err_r, self._pipe_err_w) = os.pipe()
        
        # create environment
        exec_env = dict()
        exec_env.update(os.environ)

        # copy the OS environment into our local environment
        if env is not None:
            exec_env.update(env)

        # check whether we need to attach a pipe to the standard input
        if stdin:
            _pipe_in = sp.PIPE
        else:
            _pipe_in = None
        
        # start the process
        self._process = sp.Popen(args, shell=False, env=exec_env,
                                 stdin=_pipe_in, stdout=self._pipe_out_w,
                                 stderr=self._pipe_err_w)
        
        if isinstance(stdin, six.string_types):
            # assume it is a string
            self._process.stdin.write(stdin.encode())
            self._process.stdin.close()
        
        # save information about the process
        self.pid = self._process.pid


    def __del__(self):
        os.close(self._pipe_out_r)
        os.close(self._pipe_out_w)
        os.close(self._pipe_err_r)
        os.close(self._pipe_err_w)


    def write(self, data):
        """ write data to the standard input of the process """
        if self._process.stdin is None:
            raise RuntimeError('Process was not started with `stdin=True`')
        self._process.stdin.write(data)


    def handle_stdout(self, output):
        """ callback function that is called when the program writes to its 
        stdout """
        pass


    def handle_stderr(self, output):
        """ callback function that is called when the program writes to its 
        stderr """
        pass


    def update(self, timeout=None):
        """ return whether the program wrote anything. If this is the case, the
        callbacks are called accordingly.
        
        `timeout` determines the frequency of polling the results. If it is not
            given, the value set at instantiation is used.
        """
        if timeout is None:
            timeout = self.timeout
        
        ready, _, _ = select.select([self._pipe_out_r, self._pipe_err_r], [],
                                    [], self.timeout)
        
        if ready:
            if self._pipe_out_r in ready:
                output = os.read(self._pipe_out_r, self.bufsize)
                self.handle_stdout(output.decode('utf-8'))
                
            if self._pipe_err_r in ready:
                output = os.read(self._pipe_err_r, self.bufsize)
                self.handle_stderr(output.decode('utf-8'))
                
            return True
        else:
            return False
        
        
    @property
    def alive(self):
        """ check whether the program is still running """
        return self._process.poll() is None 


    def wait(self):
        """ wait for the program to finish """
        # wait until the process is no longer alive
        while self.alive:
            self.update()
            
        # capture the remaining output if there is any
        while self.update(0):
            pass
        
        return self._process.returncode            
                
