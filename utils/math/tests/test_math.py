'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from .. import _math as m
from utils.math._math import is_equidistant



class TestMath(unittest.TestCase):
    """ test generic math functions """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    
    def test_xlog2x(self):
        """ test the xlog2x function """
        self.assertEqual(m.xlog2x(0), 0)
        self.assertEqual(m.xlog2x(0.5), 0.5*np.log2(0.5))
        self.assertEqual(m.xlog2x(1), 0)
        
        a = np.array([0, 0.5, 1])
        b = np.array([0, 0.5*np.log2(0.5), 0])
        np.testing.assert_allclose(m.xlog2x(a), b)


    def test_average_angles(self):
        """ test average_angles function """
        x = 2 * np.random.random((10, 2)) - 1
        x = x / np.linalg.norm(x, axis=1)[:, None]
        a = np.arctan2(x[:, 0], x[:, 1])
        a_mean = np.arctan2(*x.mean(axis=0))
        
        self.assertAlmostEqual(m.average_angles(2*a, period=4*np.pi), 2*a_mean)


    def test_euler_phi(self):
        """ test the euler_phi function """
        self.assertEqual(m.euler_phi(1), 1)
        self.assertEqual(m.euler_phi(36), 12)
        self.assertEqual(m.euler_phi(99), 60)
        
    
    def test_arrays_close(self):
        """ test the arrays_close function """
        a = 10 + np.random.random(10)
        self.assertFalse(m.arrays_close(a, np.arange(5)))
        
        for rtol, atol in [[1e-4, 1e-4], [1e-4, 1e-15], [1e-15, 1e-4]]:
            msg = 'Comparison failed for rtol=%1.0e, atol=%1.0e' % (rtol, atol)
            b = (1 + 0.5*rtol)*a
            self.assertTrue(m.arrays_close(a, b, rtol, atol), msg=msg)
            b = a + atol
            self.assertTrue(m.arrays_close(a, b, rtol, atol), msg=msg)
            b = (1 + 2*rtol)*a + atol
            self.assertFalse(m.arrays_close(a, b, rtol, atol), msg=msg)
            b = (1 + rtol)*a + 2 * atol
            self.assertFalse(m.arrays_close(a, b, rtol, atol), msg=msg)
        
        
    def test_logspace(self):
        """ test the logspace function """
        for num in [10, None]:
            for a, b in [[0.5, 75], [0.2, 0.2]]:
                x = m.logspace(a, b, num)
                self.assertAlmostEqual(x[0], a)
                self.assertAlmostEqual(x[-1], b)
                d = x[1:] / x[:-1]
                np.testing.assert_allclose(d, d.mean())
                
                
    def test_is_pos_semidef(self):
        """ test the is_pos_semidef function """
        self.assertTrue(m.is_pos_semidef([[1, 0], [0, 1]]))
        self.assertTrue(m.is_pos_semidef([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]))
        self.assertTrue(m.is_pos_semidef([[1, 1], [-1, 1]]))
        self.assertTrue(m.is_pos_semidef([[0, 0], [0, 0]]))
        self.assertTrue(m.is_pos_semidef([[1, 1], [1, 1]]))
        self.assertFalse(m.is_pos_semidef([[-1, 0], [0, -1]]))
        self.assertFalse(m.is_pos_semidef([[-1, 0], [1, 1]]))
            
            
    def test_trim_nan(self):
        """ test trim_nan function """
        arr = [np.nan, np.nan, 1, 2, np.nan, np.nan]
        np.testing.assert_array_equal(m.trim_nan(arr), [1, 2])
        np.testing.assert_array_equal(m.trim_nan(arr, left=False),
                                      [np.nan, np.nan, 1, 2])
        np.testing.assert_array_equal(m.trim_nan(arr, right=False),
                                      [1, 2, np.nan, np.nan])
        
        arr = [np.nan, np.nan]
        np.testing.assert_array_equal(m.trim_nan(arr), [])
        np.testing.assert_array_equal(m.trim_nan(arr, left=False), [])
        np.testing.assert_array_equal(m.trim_nan(arr, right=False), [])

        np.testing.assert_array_equal(m.trim_nan([1, np.nan], left=False), [1])
        np.testing.assert_array_equal(m.trim_nan([np.nan, 1], right=False), [1])
        
    
    def test_mean(self):
        """ test the mean function """
        x = np.random.random(10)
        self.assertAlmostEqual(m.mean(x), x.mean())
        self.assertAlmostEqual(m.mean(iter(x)), x.mean())
        
        
    def test_round(self):
        """ test the round_to_even and round_to_odd functions """
        self.assertEqual(m.round_to_even(0), 0)
        self.assertEqual(m.round_to_even(0.9), 0)
        self.assertEqual(m.round_to_even(1), 2)
        self.assertEqual(m.round_to_even(1.1), 2)
        
        self.assertEqual(m.round_to_odd(0), 1)
        self.assertEqual(m.round_to_odd(0.9), 1)
        self.assertEqual(m.round_to_odd(1), 1)
        self.assertEqual(m.round_to_odd(1.9), 1)
        self.assertEqual(m.round_to_odd(2), 3)
        
        
    def test_popcount(self):
        """ tests the popcount function """
        for x in (0, 1, 34, 1984):
            self.assertEqual(m.popcount(x), bin(x).count('1'))
            
            
    def test_is_equidistant(self):
        """ test the is_equidistant function """
        self.assertTrue(is_equidistant([]))
        self.assertTrue(is_equidistant([1]))
        self.assertTrue(is_equidistant([1, 2]))
        self.assertTrue(is_equidistant([1, 2, 3]))
        self.assertFalse(is_equidistant([1, 2, 3.001]))
    
            
    def test_contiguous_true_regions(self):
        """ tests the contiguous_true_regions function """
        self.assertListEqual(m.contiguous_true_regions([]), [])

        # test several representations of False and True
        for f in [0, False]:
            for t in [1, True]:
                res = np.array(m.contiguous_true_regions([f, t, t, f]))
                np.testing.assert_array_equal(res, np.array([(1, 3)]))
                
                res = np.array(m.contiguous_true_regions([t, f, f, t]))
                np.testing.assert_array_equal(res, np.array([(0, 1), (3, 4)]))
                    
            
    def test_contiguous_int_regions_iter(self):
        """ tests the contiguous_int_regions_iter function """
        data = [1, 1, 2, 2]
        result = [(1, 0, 2), (2, 2, 4)]
        self.assertListEqual(list(m.contiguous_int_regions_iter(data)), result)
        
        data = [1, 2, 1]
        result = [(1, 0, 1), (2, 1, 2), (1, 2, 3)]
        self.assertListEqual(list(m.contiguous_int_regions_iter(data)), result)
        
        
        
        