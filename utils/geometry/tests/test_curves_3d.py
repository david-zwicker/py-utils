'''
Created on Aug 29, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from ..curves_3d import Curve3D 



class TestCurves3D(unittest.TestCase):


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def test_trivial(self):
        """ tests trivial cases """
        # wrong input
        self.assertRaises(ValueError, lambda: Curve3D([1, 2]))
        self.assertRaises(ValueError, lambda: Curve3D([[1, 2], [3, 4, 5]]))
        
        # zero length
        self.assertEqual(Curve3D([]).length, 0)
        
        # one point
        c = Curve3D([0, 0, 0])
        self.assertEqual(c.length, 0)
        np.testing.assert_array_equal(list(c), [[0, 0, 0]])
        self.assertRaises(ValueError, lambda: list(c.iter(with_normals=True)))


    def test_simple(self):
        """ test simple 3d curves """
        ps = np.array([[0, 0, 0], [0, 0, 1]])
        c = Curve3D(ps)
        self.assertEqual(c.length, 1)
        np.testing.assert_array_equal(list(c), ps)
        np.testing.assert_array_equal(list(c.iter(with_normals=False)), ps)
        
        for k, (p, n) in enumerate(c.iter(with_normals=True)):
            np.testing.assert_array_equal(p, ps[k])
            np.testing.assert_array_equal(n, ps[1] - ps[0])



if __name__ == "__main__":
    unittest.main()
