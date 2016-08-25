'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

from .._math import *  # @UnusedWildImport



class TestMath(unittest.TestCase):
    """ test generic math functions """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    
    def test_xlog2x(self):
        """ test the xlog2x function """
        self.assertEqual(xlog2x(0), 0)
        self.assertEqual(xlog2x(0.5), 0.5*np.log2(0.5))
        self.assertEqual(xlog2x(1), 0)
        
        a = np.array([0, 0.5, 1])
        b = np.array([0, 0.5*np.log2(0.5), 0])
        np.testing.assert_allclose(xlog2x(a), b)


    def test_euler_phi(self):
        """ test the euler_phi function """
        self.assertEqual(euler_phi(1), 1)
        self.assertEqual(euler_phi(36), 12)
        self.assertEqual(euler_phi(99), 60)
        
        
    def test_logspace(self):
        """ test the logspace function """
        for a, b in [[0.5, 75], [0.2, 0.2]]:
            x = logspace(a, b, 10)
            self.assertAlmostEqual(x[0], a)
            self.assertAlmostEqual(x[-1], b)
            d = x[1:] / x[:-1]
            np.testing.assert_allclose(d, d.mean())
            
    
    def test_mean(self):
        """ test the mean function """
        x = np.random.random(10)
        self.assertAlmostEqual(mean(x), x.mean())
        self.assertAlmostEqual(mean(iter(x)), x.mean())
        
        
    def test_round(self):
        """ test the round_to_even and round_to_odd functions """
        self.assertEqual(round_to_even(0), 0)
        self.assertEqual(round_to_even(0.9), 0)
        self.assertEqual(round_to_even(1), 2)
        self.assertEqual(round_to_even(1.1), 2)
        
        self.assertEqual(round_to_odd(0), 1)
        self.assertEqual(round_to_odd(0.9), 1)
        self.assertEqual(round_to_odd(1), 1)
        self.assertEqual(round_to_odd(1.9), 1)
        self.assertEqual(round_to_odd(2), 3)
        
        
    def test_popcount(self):
        """ tests the popcount function """
        for x in (0, 1, 34, 1984):
            self.assertEqual(popcount(x), bin(x).count('1'))
        
        