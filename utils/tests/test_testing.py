'''
Created on Aug 29, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

from .. import testing



class TestMisc(unittest.TestCase):


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def test_repeat(self):
        """ test the repeat decorator class """
                
        @testing.repeat(5)
        def inc_counter(cnt):
            cnt[0] += 1
        
        counter = [0]  # use mutable container to retain data
        inc_counter(counter)
        self.assertEqual(counter, [5])
    


if __name__ == "__main__":
    unittest.main()
