'''
Created on Nov 8, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from .. import shapes_nd


class TestShapesND(unittest.TestCase):

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def test_line_2d(self):
        """ test the Line class in 2d """
        line = shapes_nd.Line.from_points([0, 0], [0, 1])
        np.testing.assert_almost_equal(line.project_point([1, 1]), [0, 1])
        np.testing.assert_almost_equal(line.project_point([-1, -2]), [0, -2])

        line = shapes_nd.Line.from_points([0, 0], [1, 1])
        np.testing.assert_almost_equal(line.project_point([[1, 0], [-2, -2]]),
                                       [[0.5, 0.5], [-2, -2]])


    def test_line(self):
        """ tests the Line class """
        # test simple nd case
        dim = np.random.randint(4, 6)
        origin = np.random.randn(dim)
        direction = np.random.randn(dim)
        line = shapes_nd.Line(origin, direction)

        self.assertIsInstance(repr(line), str)
        self.assertEqual(line.dim, dim)
        p = np.random.randn(dim)
        self.assertTrue(line.contains_point(line.project_point(p)))
        ps = np.random.randn(5, dim)
        self.assertTrue(np.all(line.contains_point(line.project_point(ps))))
        
        # test wrong arguments
        self.assertRaises(ValueError, lambda: shapes_nd.Line([], [1]))
        self.assertRaises(ValueError, lambda: shapes_nd.Line([1], [1, 2]))
        

    def test_plane_2d(self):
        """ test the Plane class in 2d """
        plane1 = shapes_nd.Plane([0, 0], [0, 1])
        np.testing.assert_almost_equal(plane1.project_point([1, 1]), [1, 0])
        np.testing.assert_almost_equal(plane1.project_point([-1, -2]), [-1, 0])
        np.testing.assert_almost_equal(plane1.flip_normal().normal, [0, -1])

        plane2 = shapes_nd.Plane([2, 2], [2, 0])
        np.testing.assert_almost_equal(plane2.project_point([[1, 0], [-2, -2]]),
                                       [[2, 0], [2, -2]])
        self.assertNotEqual(plane1, plane2)
        
        # test creating random plane
        ps = np.random.randn(2, 2)
        plane3 = shapes_nd.Plane.from_points(ps)
        self.assertTrue(np.all(plane3.contains_point(ps)))

        ps = [[0, 0], [1, 1], [2, 2]]
        plane4 = shapes_nd.Plane.from_points(ps)
        self.assertTrue(np.all(plane4.contains_point(ps)))

        ps = [[0, 0], [1, 1], [2, 2. + 1e-10]]
        plane5 = shapes_nd.Plane.from_points(ps)
        np.testing.assert_almost_equal(plane5.distance_point(ps), 0)
        
        self.assertRaises(ValueError,
                          lambda: shapes_nd.Plane.from_points([[1, 2]]))
        
        plane6 = shapes_nd.Plane.from_average([plane1])
        self.assertEqual(plane1, plane6)
        plane_t = shapes_nd.Plane([0, 2], [0, 1])
        plane7 = shapes_nd.Plane.from_average([plane1, plane_t])
        np.testing.assert_almost_equal(plane7.origin, [0, 1])
        np.testing.assert_almost_equal(plane7.normal, [0, 1])


    def test_plane(self):
        """ tests the Plane class """
        # test simple nd case
        dim = np.random.randint(4, 6)
        origin = np.random.randn(dim)
        normal = np.random.randn(dim)
        plane = shapes_nd.Plane(origin, normal)
        np.testing.assert_almost_equal(plane.flip_normal().normal, -normal)
        
        self.assertIsInstance(repr(plane), str)
        self.assertEqual(plane.dim, dim)
        p = np.random.randn(dim)
        self.assertTrue(plane.contains_point(plane.project_point(p)))
        ps = np.random.randn(5, dim)
        self.assertTrue(np.all(plane.contains_point(plane.project_point(ps))))
        
        # test wrong arguments
        self.assertRaises(ValueError, lambda: shapes_nd.Plane([], [1]))
        self.assertRaises(ValueError, lambda: shapes_nd.Plane([1], [1, 2]))



if __name__ == "__main__":
    unittest.main()
