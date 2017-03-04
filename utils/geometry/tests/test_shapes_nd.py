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
        
        self.assertAlmostEqual(line.point_distance([0, 0]), 0)
        self.assertAlmostEqual(line.point_distance([0, 4]), 0)
        self.assertAlmostEqual(line.point_distance([1, 0]), 1)
        self.assertAlmostEqual(line.point_distance([1, 5]), 1)
        np.testing.assert_almost_equal(line.point_distance([[0, 0], [0, 1]]),
                                       [0, 0])
        
        l2 = shapes_nd.Line(np.random.random(2), np.random.random(2))
        self.assertAlmostEqual(line.distance(l2), 0)
        l2 = shapes_nd.Line([1, 1], [0, 1])
        self.assertAlmostEqual(line.distance(l2), 1)
        l2 = shapes_nd.Line([-1, 1], [0, 1])
        self.assertAlmostEqual(line.distance(l2), 1)

        line = shapes_nd.Line.from_points([0, 0], [1, 1])
        np.testing.assert_almost_equal(line.project_point([[1, 0], [-2, -2]]),
                                       [[0.5, 0.5], [-2, -2]])
        self.assertAlmostEqual(line.point_distance([2, 2]), 0)
        self.assertAlmostEqual(line.point_distance([0, 1]), 1/np.sqrt(2))
        
        
    def test_line_3d(self):
        """ test the Line class in 2d """
        # example taken from http://math.stackexchange.com/q/210848/198991
        l1 = shapes_nd.Line([-1, 1, 4], [1, 1, -1])
        l2 = shapes_nd.Line([5, 3, -3], [-2, 0, 1])
        self.assertAlmostEqual(l1.distance(l2), np.sqrt(6))


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
        self.assertAlmostEqual(line.point_distance(origin), 0)
        self.assertAlmostEqual(line.point_distance(origin + 2*direction), 0)
        l2 = shapes_nd.Line(np.random.random(dim), np.random.random(dim))
        self.assertGreater(line.distance(l2), 0)
        
        # test wrong arguments
        self.assertRaises(ValueError, lambda: shapes_nd.Line([], [1]))
        self.assertRaises(ValueError, lambda: shapes_nd.Line([1], [1, 2]))
        

    def test_segment_2d(self):
        """ test the Segment class in 2d """
        segment = shapes_nd.Segment([0, 0], [0, 1])

        self.assertIsInstance(repr(segment), str)
        self.assertEqual(segment.dim, 2)
        self.assertAlmostEqual(segment.length, 1)
        np.testing.assert_equal(segment.points, np.array([[0, 0], [0, 1]]))


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
        np.testing.assert_almost_equal(plane.flip_normal().normal,
                                       -plane.normal)
        
        self.assertIsInstance(repr(plane), str)
        self.assertEqual(plane.dim, dim)
        p = np.random.randn(dim)
        self.assertTrue(plane.contains_point(plane.project_point(p)))
        ps = np.random.randn(5, dim)
        self.assertTrue(np.all(plane.contains_point(plane.project_point(ps))))
        
        # test wrong arguments
        self.assertRaises(ValueError, lambda: shapes_nd.Plane([], [1]))
        self.assertRaises(ValueError, lambda: shapes_nd.Plane([1], [1, 2]))
        
        
    def test_plane_point_distance(self):
        """ test the distance calculation """ 
        plane = shapes_nd.Plane([0, 0, 0], [1, 0, 0])
        for oriented in (True, False):
            for d in np.arange(-2, 3):
                dist = plane.distance_point([d, 0, 0], oriented=oriented)
                self.assertAlmostEqual(dist, d if oriented else np.abs(d))
                dist = plane.distance_point([0, d, 0], oriented=oriented)
                self.assertAlmostEqual(dist, 0)

        plane = shapes_nd.Plane([0, 0], [1, 1])
        self.assertAlmostEqual(plane.distance_point([1, 1]), np.sqrt(2))
        self.assertAlmostEqual(plane.distance_point([-1, -1]), np.sqrt(2))
        dist = plane.distance_point([-2, -2], oriented=True)
        self.assertAlmostEqual(dist, -2 * np.sqrt(2))
        

    def test_cuboid_2d(self):
        """ test Cuboid class in 2d """
        c = shapes_nd.Cuboid([-1, -1], [2, 2])
        self.assertEqual(c.dim, 2)
        self.assertEqual(c.volume, 4)
        np.testing.assert_array_equal(c.centroid, [0, 0])

        c1 = c.buffer(1, inplace=False)
        self.assertEqual(c1.volume, 16)
        c2 = c.extend([1, 1], 1, inplace=False)
        self.assertEqual(c2.volume, 9)
        np.testing.assert_array_equal(c2.bounds, [[-1, 2], [-1, 2]])
        c2.extend([-1, -1], 1)
        self.assertEqual(c1, c2)
        

    def test_cylinder(self):
        """ test the Cylinder class """
        o = np.random.random(3)  # random origin
        cyl = shapes_nd.Cylinder(o + [0, 0, 0], o + [0, 0, 10], 2)
        
        self.assertIsInstance(repr(cyl), str)
        self.assertEqual(cyl.height, 10)
        self.assertEqual(cyl.dim, 3)
        
        self.assertEqual(cyl.distance_point(o + [1, 0, 0]), 0)
        self.assertEqual(cyl.distance_point(o + [2, 0, 0]), 0)
        self.assertEqual(cyl.distance_point(o + [3, 0, 0]), 1)
        self.assertEqual(cyl.distance_point(o + [2, 0, -2]), 2)
        self.assertEqual(cyl.distance_point(o + [2, 0, 12]), 2)
        self.assertEqual(cyl.distance_point(o + [5, 0, -4]), 5)
        self.assertEqual(cyl.distance_point(o + [5, 0, 14]), 5)
        self.assertEqual(cyl.distance_point(o + [0, 5, 14]), 5)


    def test_asanyarray_flags(self):
        """ test the asanyarray_flags function """
        self.assertIsNot(np.arange(3), shapes_nd.asanyarray_flags(range(3)))
        
        a = np.random.random(3).astype(np.double)
        self.assertIs(a, shapes_nd.asanyarray_flags(a))
        self.assertIs(a, shapes_nd.asanyarray_flags(a, np.double))
        self.assertIs(a, shapes_nd.asanyarray_flags(a, writeable=True))
        self.assertIsNot(a, shapes_nd.asanyarray_flags(a, np.int))
        self.assertIsNot(a, shapes_nd.asanyarray_flags(a, writeable=False))
        
        for dtype in (np.int, np.double):
            b = shapes_nd.asanyarray_flags(a, dtype)
            self.assertEqual(b.dtype, dtype)

        for writeable in (True, False):
            b = shapes_nd.asanyarray_flags(a, writeable=writeable)
            self.assertEqual(b.flags.writeable, writeable)



if __name__ == "__main__":
    unittest.main()
