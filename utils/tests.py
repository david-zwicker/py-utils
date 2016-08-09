'''
Created on Aug 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np
from six.moves import zip_longest

from .math import arrays_close
      

      
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
      
            

if __name__ == '__main__':
    unittest.main()
