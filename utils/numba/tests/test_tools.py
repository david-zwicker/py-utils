'''
Created on Aug 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from ...math import distributions
from .. import tools

      
     
class TestNumbaTools(unittest.TestCase):
    """ unit tests for the numba tools """

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel
    
    
    def test_numba_stats(self):
        """ test the numba implementation of statistics functions """
        if not tools.numba:
            return  # numba was not found
        
        for _ in range(10):
            mean = np.random.random() + 0.1
            var = np.random.random() + 0.1
            x = np.random.random() + 0.1
            dist_LN = distributions.lognorm_mean_var(mean, var)
            self.assertAlmostEqual(dist_LN.pdf(x),
                                   tools.lognorm_pdf(x, mean, var))
            self.assertAlmostEqual(dist_LN.cdf(x),
                                   tools.lognorm_cdf(x, mean, var))
    
    

if __name__ == '__main__':
    unittest.main()
