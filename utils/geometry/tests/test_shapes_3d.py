'''
Created on Jul 5, 2017

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from .. import shapes_3d



class TestShapes3D(unittest.TestCase):

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def test_CoordinatePlane(self):
        """ test the CoordinatePlane class """
        origin = np.random.randn(3)
        normal = np.random.randn(3)
        up_vector = np.random.randn(3)
        plane = shapes_3d.CoordinatePlane(origin, normal, up_vector)
        
        p3 = [0, 1, 0]
        c, d = plane.project_point(p3, ret_dist=True)
        np.testing.assert_almost_equal(p3, plane.revert_projection(c, d))
        p3 = np.random.randn(5, 3)
        c, d = plane.project_point(p3, ret_dist=True)
        np.testing.assert_almost_equal(p3, plane.revert_projection(c, d))



if __name__ == "__main__":
    unittest.main()
