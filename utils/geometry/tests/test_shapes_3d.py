'''
Created on Jul 5, 2017

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest
import pickle

import numpy as np

from .. import shapes_3d, shapes_nd



class TestShapes3D(unittest.TestCase):

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def test_CoordinatePlane(self):
        """ test the CoordinatePlane class """
        origin = np.random.randn(3)
        normal = np.random.randn(3)
        up_vector = np.random.randn(3)
        plane = shapes_nd.Plane(origin, normal)
        cplane = shapes_3d.CoordinatePlane(origin, normal, up_vector)
        
        np.testing.assert_almost_equal(cplane.dim, plane.dim)
        np.testing.assert_almost_equal(cplane.origin, plane.origin)
        np.testing.assert_almost_equal(cplane.normal, plane.normal)
        
        p3 = [0, 1, 0]
        c, d = cplane.project_point(p3, ret_dist=True)
        np.testing.assert_almost_equal(p3, cplane.revert_projection(c, d))
        p3 = np.random.randn(5, 3)
        c, d = cplane.project_point(p3, ret_dist=True)
        np.testing.assert_almost_equal(p3, cplane.revert_projection(c, d))


    def test_pickle(self):
        """ test whether the objects can be pickled and unpickled """
        origin = np.random.randn(3)
        normal = np.random.randn(3)
        up_vector = np.random.randn(3)
        plane = shapes_3d.CoordinatePlane(origin, normal, up_vector)
        
        p2 = pickle.loads(pickle.dumps(plane))
        np.testing.assert_almost_equal(plane.origin, p2.origin)
        np.testing.assert_almost_equal(plane.normal, p2.normal)
        np.testing.assert_almost_equal(plane.basis_u, p2.basis_u)
        np.testing.assert_almost_equal(plane.basis_v, p2.basis_v)


if __name__ == "__main__":
    unittest.main()
