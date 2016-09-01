'''
Created on Aug 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging
import sys
import unittest
import warnings
from contextlib import contextmanager

import numpy as np
from six.moves import zip_longest

from .math import arrays_close
      


class MockLoggingHandler(logging.Handler):
    """ Mock logging handler to check for expected logs.

    Messages are available from an instance's ``messages`` dict, in order,
    indexed by a lowercase log level string (e.g., 'debug', 'info', etc.).
    
    Adapted from http://stackoverflow.com/a/20553331/932593
    """

    def __init__(self, *args, **kwargs):
        self.messages = {'debug': [], 'info': [], 'warning': [], 'error': [],
                         'critical': []}
        super(MockLoggingHandler, self).__init__(*args, **kwargs)


    def emit(self, record):
        """ 
        Store a message from ``record`` in the instance's ``messages`` dict.
        """
        self.acquire()
        try:
            self.messages[record.levelname.lower()].append(record.getMessage())
        finally:
            self.release()


    def reset(self):
        """ reset all messages """
        self.acquire()
        try:
            for message_list in self.messages.values():
                message_list.clear()
        finally:
            self.release()
            

      
class TestBase(unittest.TestCase):
    """ extends the basic TestCase class with some convenience functions """ 
      
    def assertAllClose(self, arr1, arr2, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the arrays a and b """
        try:
            # try to convert to numpy arrays
            arr1 = np.asanyarray(arr1)
            arr2 = np.asanyarray(arr2)
            
        except ValueError:
            # try iterating explicitly
            try:
                for v1, v2 in zip_longest(arr1, arr2):
                    self.assertAllClose(v1, v2, rtol, atol, msg)
            except TypeError:
                if msg is None:
                    msg = ""
                else:
                    msg += "; "
                raise TypeError(msg + "Don't know how to compare %s and %s"
                                % (arr1, arr2))
                
        else:
            if msg is None:
                msg = 'Values are not equal'
            msg += '\n%s !=\n%s)' % (arr1, arr2)
            is_close = arrays_close(arr1, arr2, rtol, atol, equal_nan=True)
            self.assertTrue(is_close, msg)

        
    def assertDictAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the dictionaries a and b """
        if msg is None:
            msg = ''
        else:
            msg += '\n'
        
        for k, v in a.items():
            # create a message if non was given
            submsg = msg + ('Dictionaries differ for key `%s` (%s != %s)'
                            % (k, v, b[k]))
                
            # try comparing as numpy arrays and fall back if that doesn't work
            try:
                self.assertAllClose(v, b[k], rtol, atol, submsg)
            except TypeError:
                self.assertEqual(v, b[k], submsg)
      
            
            
class WarnAssertionsMixin(object):
    """
    Mixing that allows to test for warnings
    Code inspired by https://blog.ionelmc.ro/2013/06/26/testing-python-warnings/
    """
    
    @contextmanager
    def assertNoWarnings(self):
        try:
            warnings.simplefilter("error")
            yield
        finally:
            warnings.resetwarnings()


    @contextmanager
    def assertWarnings(self, messages):
        """
        Asserts that the given messages are issued in the given order.
        """
        if not messages:
            raise RuntimeError("Use assertNoWarnings instead!")

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            for mod in sys.modules.values():
                if hasattr(mod, '__warningregistry__'):
                    mod.__warningregistry__.clear()
            yield
            warning_list = [w.message.args[0] for w in warning_list]
            for message in messages:
                if not any(message in warning for warning in warning_list):
                    self.fail('Message `%s` was not contained in warnings'
                              % message)
            


if __name__ == '__main__':
    unittest.main()
