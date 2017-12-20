'''
Created on Aug 26, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import itertools
import unittest
import os.path
import time

from .. import concurrency



class TestConcurrency(unittest.TestCase):


    def _test_WorkerThread(self, use_threads=True, synchronous=True):
        """ test the WorkerThread class with a single set of parameters """
        def calculate(arg):
            return arg
        
        worker_thread = concurrency.WorkerThread(calculate, use_threads,
                                                 synchronous)
        
        for arg in [1, None, [1, 2]]:
            worker_thread.put(arg)
            time.sleep(0.001)  # wait a bit until the thread finished
            self.assertEqual(worker_thread.get(), arg)


    def test_WorkerThread(self):
        """ test the WorkerThread class """
        for use_threads in (True, False):
            for synchronous in (True, False):
                self._test_WorkerThread(use_threads, synchronous)


    def test_MonitorProcessOutput(self):
        """ test the MonitorProcessOutput class """
        this_dir = os.path.dirname(__file__)
        script_path = os.path.join(this_dir, 'resources', 'child_job.py')
        
        class TestMonitor(concurrency.MonitorProcessOutput):
            def __init__(self, count, **kwargs):
                super(TestMonitor, self).__init__([script_path, str(count)],
                                                  **kwargs)
                self.stdout = ''
                self.stderr = ''
                
            def handle_stdout(self, output):
                self.stdout += output

            def handle_stderr(self, output):
                self.stderr += output
                
        for bufsize, timeout in itertools.product((1, 1014), (0, 0.1)):
            test = TestMonitor(4, timeout=timeout, bufsize=bufsize)
            self.assertEqual(test.alive, True)
            rc = test.wait()
            self.assertEqual(test.alive, False)
            self.assertEqual(rc, 0)
            self.assertEqual(test.stdout, '1\n123\n')
            self.assertEqual(test.stderr, 'AB\nABCD\n')
        


if __name__ == "__main__":
    unittest.main()
