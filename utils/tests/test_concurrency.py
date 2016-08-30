'''
Created on Aug 26, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest
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



if __name__ == "__main__":
    unittest.main()
