'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import sys
import unittest

import numpy as np

from .. import _math as m
from utils.math._math import diff1d_circular



class TestMath(unittest.TestCase):
    """ test generic math functions """

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel
    
    def test_xlog2x(self):
        """ test the xlog2x function """
        self.assertEqual(m.xlog2x(0), 0)
        self.assertEqual(m.xlog2x(0.5), 0.5*np.log2(0.5))
        self.assertEqual(m.xlog2x(1), 0)
        
        a = np.array([0, 0.5, 1])
        b = np.array([0, 0.5*np.log2(0.5), 0])
        np.testing.assert_allclose(m.xlog2x(a), b)


    def test_heaviside(self):
        """ test the heaviside function """
        self.assertEqual(m.heaviside(-1), 0)
        self.assertEqual(m.heaviside(0), 0.5)
        self.assertEqual(m.heaviside(1), 1)
        self.assertEqual(m.heaviside(-np.inf), 0)
        self.assertTrue(np.isnan(m.heaviside(np.nan)))
        self.assertEqual(m.heaviside(np.inf), 1)
        
        np.testing.assert_array_equal(m.heaviside(np.array([-1, 0, 1])),
                                      np.array([0, 0.5, 1]))
        
        a = np.array([-1, 0, 1])
        m.heaviside(a, out=a)
        np.testing.assert_array_equal(a, np.array([0, 0, 1]))

        a = np.array([-1, 0, 1], np.double)
        m.heaviside(a, out=a)
        np.testing.assert_array_equal(a, np.array([0, 0.5, 1]))


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
        self.assertFalse(m.arrays_close(a, np.arange(10)))
        
        for rtol, atol in [[1e-4, 1e-4], [1e-4, 1e-15], [1e-15, 1e-4]]:
            msg = 'Comparison failed for rtol=%1.0e, atol=%1.0e' % (rtol, atol)
            b = (1 + 0.5*rtol)*a
            self.assertTrue(m.arrays_close(a, b, rtol, atol), msg=msg)
            b = a + atol
            self.assertTrue(m.arrays_close(a, b, rtol, atol), msg=msg)
            b = (1 + 2*rtol)*a + atol
            self.assertFalse(m.arrays_close(a, b, rtol, atol), msg=msg)
            b = (1 + 1.1*rtol)*(a + 2*atol)
            self.assertFalse(m.arrays_close(a, b, rtol, atol), msg=msg)
            
        # test special cases
        self.assertFalse(m.arrays_close([0], [np.nan], equal_nan=True))
        self.assertFalse(m.arrays_close([0], [np.nan], equal_nan=False))
        self.assertTrue(m.arrays_close([np.nan], [np.nan], equal_nan=True))
        self.assertFalse(m.arrays_close([np.nan], [np.nan], equal_nan=False))
        self.assertTrue(m.arrays_close([0, np.nan], [0, np.nan],
                                       equal_nan=True))
        self.assertFalse(m.arrays_close([0, np.nan], [0, np.nan],
                                        equal_nan=False))
        self.assertTrue(m.arrays_close([np.inf], [np.inf]))
        self.assertFalse(m.arrays_close([-np.inf], [np.inf]))
        
        
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
        
        
    def test_diff1d_circular(self):
        """ test the diff1d_circular function """
        for dtype in (np.int, np.uint, np.double):
            x = np.array([0, 1, 2], dtype)
            np.testing.assert_equal(diff1d_circular(x, 3), np.ones(3, np.int))
            np.testing.assert_equal(diff1d_circular(x, 3.5), np.r_[1, 1, 1.5])
            np.testing.assert_equal(diff1d_circular(x, 4), np.r_[1, 1, -2])
            np.testing.assert_equal(diff1d_circular(x, 4.5), np.r_[1, 1, -2])
            
        self.assertRaises(TypeError, lambda: diff1d_circular(np.array('s'), 1))
        
    
    def test_mean(self):
        """ test the mean function """
        x = np.random.random(10)
        self.assertAlmostEqual(m.mean(x), x.mean())
        self.assertAlmostEqual(m.mean(iter(x)), x.mean())
        
        
    def test_moving_average(self):
        """ test the moving_average function """
        np.testing.assert_allclose(m.moving_average([]), [])
        
        arr = np.arange(4)
        np.testing.assert_allclose(m.moving_average(arr, 2), [0.5, 1.5, 2.5])
        np.testing.assert_allclose(m.moving_average(arr, 3), [1, 2])

        arr = np.random.random((10, 10))
        res1 = m.moving_average(arr)
        res2 = np.apply_along_axis(m.moving_average, axis=0, arr=arr)
        np.testing.assert_allclose(res1, res2)
        
        arr = np.random.random(10)
        np.testing.assert_allclose(m.moving_average(arr, 1), arr)
        np.testing.assert_allclose(m.moving_average(arr, len(arr)),
                                   [arr.mean()])
        
        
    def test_Interpolate_1D_Extrapolated(self):
        """ test the Interpolate_1D_Extrapolated class """
        interp = m.Interpolate_1D_Extrapolated([0, 1], [0, 1])
        self.assertEqual(interp(-1), 0)
        self.assertEqual(interp(0), 0)
        self.assertEqual(interp(0.5), 0.5)
        self.assertEqual(interp(1), 1)
        self.assertEqual(interp(2), 1)
        np.testing.assert_allclose(interp([-0.1, 0, 0.5, 1, 1.1]),
                                   [0, 0, 0.5, 1, 1])
        np.testing.assert_allclose(interp(np.array([-0.1, 0, 0.5, 1, 1.1])),
                                   [0, 0, 0.5, 1, 1])
                                   
        
        
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
        
    
    def test_calc_entropy(self):
        """ test the calc_entropy function """
        entropy_functions = (m._ENTROPY_FUNCTIONS + [m.calc_entropy])
        
        if sys.version_info.major > 2:
            entropy_functions.remove(m._entropy_counter1)
        
        for entropy in entropy_functions:
            msg = 'Entropy function `%s` failed' % entropy.__name__
            self.assertEqual(entropy([]), 0, msg=msg)
            self.assertEqual(entropy([1]), 0, msg=msg)
            self.assertEqual(entropy([1, 1]), 0, msg=msg)
            self.assertAlmostEqual(entropy([0, 1]), 1, msg=msg)
            self.assertAlmostEqual(entropy([0, 0.5, 1]), np.log2(3), msg=msg)
        
        
    def test_popcount(self):
        """ tests the popcount function """
        for x in (0, 1, 34, 1984):
            self.assertEqual(m.popcount(x), bin(x).count('1'))
            
            
    def test_take_popcount(self):
        """ test the take_popcount function """
        self.assertEqual(m.take_popcount([], 0), []) 
        self.assertEqual(m.take_popcount([], 1), []) 
        self.assertEqual(m.take_popcount([3, 4], 0), [3]) 
        self.assertEqual(m.take_popcount([3, 4], 1), [4]) 
        self.assertEqual(m.take_popcount([3, 4, 5, 6], 1), [4, 5]) 
        self.assertEqual(m.take_popcount([3, 4, 5, 6], 2), [6]) 
            
    
    def test_to_array(self):
        """ test the to_array function """
        a = np.arange(2)
        np.testing.assert_array_equal(a, m.to_array(a)) 
        np.testing.assert_array_equal(a, m.to_array([0, 1]))
        np.testing.assert_array_equal(a, m.to_array((0, 1)))
        np.testing.assert_array_equal(a, m.to_array(iter((0, 1))))
        np.testing.assert_array_equal(a, m.to_array(sorted((0, 1))))
        d = {0: 0, 1: 1}
        np.testing.assert_array_equal(a, m.to_array(d.keys()))
        np.testing.assert_array_equal(a, m.to_array(d.values()))
            
            
    def test_get_number_range(self):
        """ test the get_number_range function """
        # test unsupported types
        from decimal import Decimal
        self.assertRaises(TypeError, lambda: m.get_number_range(Decimal(1)))
        self.assertRaises(TypeError, lambda: m.get_number_range(np.str))
        
        # test integer ranges
        self.assertEqual(m.get_number_range(np.uint8), (0, 255))
        self.assertEqual(m.get_number_range(np.int8), (-128, 127))
        self.assertEqual(m.get_number_range(np.uint16), (0, 65535))
        self.assertEqual(m.get_number_range(np.int16), (-32768, 32767))
        self.assertEqual(m.get_number_range(np.uint32), (0, 4294967295))
        self.assertEqual(m.get_number_range(np.int32),
                         (-2147483648, 2147483647))
        self.assertEqual(m.get_number_range(np.uint64),
                         (0, 18446744073709551615))
        self.assertEqual(m.get_number_range(np.int64),
                         (-9223372036854775808, 9223372036854775807))
        
        # test float ranges
        np.testing.assert_allclose(m.get_number_range(np.single),
                                   (-3.4028235e+38, 3.4028235e+38))
        np.testing.assert_allclose(m.get_number_range(np.double),
                                   (-1.7976931348623157e+308,
                                    1.7976931348623157e+308))
            
            
    def test_homogenize_arraylist(self):
        """ test the homogenize_arraylist function """
        self.assertEqual(m.homogenize_arraylist([]), [])
        self.assertRaises(TypeError, lambda: m.homogenize_arraylist([1]))
        self.assertEqual(m.homogenize_arraylist([[1]]), np.array([[1]]))
        np.testing.assert_array_equal(m.homogenize_arraylist([[1, 2]]),
                                      np.array([[1, 2]]))
        np.testing.assert_array_equal(m.homogenize_arraylist([[1, 2], [3, 4]]),
                                      np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(m.homogenize_arraylist([[1., 2], [3]]),
                                      np.array([[1, 2], [3, np.nan]]))
        np.testing.assert_array_equal(m.homogenize_arraylist([[1.], [3, 4]]),
                                      np.array([[1, np.nan], [3, 4]]))
            
            
    def test_homogenize_unit_array(self):
        """ test the homogenize_unit_array function """
        try:
            from pint import UnitRegistry
        except ImportError:
            # the unit library was not found and we thus don't run the test
            return
        
        ureg = UnitRegistry()
        cm = ureg.centimeter
        mm = ureg.millimeter
        
        self.assertEqual(m.homogenize_unit_array([]), [])
        self.assertEqual(m.homogenize_unit_array([1]), [1])
        self.assertEqual(m.homogenize_unit_array([1] * cm), [1] * cm)
        np.testing.assert_array_equal(m.homogenize_unit_array([1, 2] * cm),
                                      [1, 2] * cm)
        np.testing.assert_array_equal(m.homogenize_unit_array([1*cm, 20*mm]),
                                      [1, 2] * cm)
        res = m.homogenize_unit_array([1*cm, 20*mm], unit=mm)
        np.testing.assert_array_equal(res, [10, 20] * mm)
        
            
    def test_is_equidistant(self):
        """ test the is_equidistant function """
        self.assertTrue(m.is_equidistant([]))
        self.assertTrue(m.is_equidistant([1]))
        self.assertTrue(m.is_equidistant([1, 2]))
        self.assertTrue(m.is_equidistant([1, 2, 3]))
        self.assertFalse(m.is_equidistant([1, 2, 3.001]))
    
            
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
        
        
    def test_safe_typecast(self):
        """ test the safe_typecast function """
        for dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8,
                      np.uint16, np.uint32]:
            iinfo = np.iinfo(dtype)
            
            val = m.safe_typecast(int(iinfo.min) - 1, dtype)
            self.assertEqual(val, iinfo.min)
            self.assertEqual(val.dtype, dtype)
            
            val = m.safe_typecast(int(iinfo.max) + 1, dtype)
            self.assertEqual(val, iinfo.max)
            self.assertEqual(val.dtype, dtype)
    
    
    
if __name__ == "__main__":
    unittest.main()
