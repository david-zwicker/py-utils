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
        self.assertRaises(ValueError, lambda: Curve3D([[1], [2]]))
        self.assertRaises(ValueError, lambda: Curve3D([[1, 2], [3, 4, 5]]))
        
        # zero length
        self.assertEqual(Curve3D([]).length, 0)
        
        # one point
        c = Curve3D([0, 0, 0])
        self.assertEqual(c.length, 0)
        np.testing.assert_array_equal(list(c), [[0, 0, 0]])
        self.assertRaises(ValueError, lambda: list(c.iter(data=['tangent'])))


    def test_simple(self):
        """ test simple 3d curves """
        ps = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        c = Curve3D(ps)
        self.assertEqual(c.length, 2)
        np.testing.assert_array_equal(list(c), ps)
        np.testing.assert_array_equal(list(c.iter()), ps)
        
        # test interpolation
        zs = np.linspace(0, 2, 7)
        for z in zs:
            np.testing.assert_array_equal(c.get_points(z), [0, 0, z])
        seq = np.c_[np.zeros(7), np.zeros(7), zs]
        np.testing.assert_array_equal(c.get_points(zs), seq)
        
        # test determining extra values
        for k, (p, d) in enumerate(c.iter(data='all')):
            np.testing.assert_array_equal(p, ps[k])
            np.testing.assert_array_equal(d['tangent'], [0, 0, 1])
            self.assertAlmostEqual(d['arc_length'], k)
            # treat end points differently
            f = 0.5 if k == 0 or k == c.num_points - 1 else 1
            self.assertAlmostEqual(d['stretching_factor'], f)
            self.assertTrue(np.isnan(d['torsion']))
            
        # test intersections with plane
        p = c.plane_intersects((0, 0, 0.5), (0, 0, 1))
        np.testing.assert_array_equal(p, [[0, 0, 0.5]])
        p = c.plane_intersects((0, 0, 0.5), np.random.rand(3))
        np.testing.assert_array_equal(p, [[0, 0, 0.5]])
        p = c.plane_intersects((0, 0.5, 0), np.ones(3))
        np.testing.assert_array_equal(p, [[0, 0, 0.5]])
            
            
    def test_circle(self):
        """ test circle """
        for smoothing in (0, 0.01):
            r = 2
            a = np.linspace(0, 2 * np.pi, 256)
            ps = np.c_[r * np.sin(a), r * np.cos(a), np.zeros_like(a)]
            c = Curve3D(ps, smoothing)
            
            self.assertAlmostEqual(c.length, 2 * np.pi * r, places=2)
            self.assertAlmostEqual(c.length, c.stretching_factors.sum())
            np.testing.assert_array_equal(list(c), ps)
                
            for b in np.linspace(0, 2 * np.pi, 37)[:-1]:
                np.testing.assert_allclose(c.get_points(r * b),
                                           [r * np.sin(b), r * np.cos(b), 0],
                                           atol=1e-3)
            
            # check the individual values
            for k, (p, d) in enumerate(c.iter(data='all')):
                np.testing.assert_array_equal(p, ps[k])
                np.testing.assert_allclose(d['tangent'],
                                           [np.cos(a[k]), -np.sin(a[k]), 0],
                                           atol=2e-2)
                np.testing.assert_allclose(d['normal'],
                                           [-np.sin(a[k]), -np.cos(a[k]), 0],
                                           atol=2e-2)
                np.testing.assert_allclose(d['binormal'], [0, 0, -1])
                self.assertAlmostEqual(d['arc_length'], r*a[k], 3)
                
                # treat end points differently
                f = 0.5 if k == 0 or k == len(a) - 1 else 1
                self.assertAlmostEqual(d['stretching_factor'],
                                       2*np.pi*r*f / (len(a) - 1), 5)
            
                self.assertAlmostEqual(d['torsion'], 0)

            # check the unit_vector system
            for k, (p, d) in enumerate(c.iter(data=['unit_vectors'])):
                np.testing.assert_array_equal(p, ps[k])
                uv = [[ np.cos(a[k]), -np.sin(a[k]),  0],
                      [-np.sin(a[k]), -np.cos(a[k]),  0],
                      [            0,             0, -1]]
                np.testing.assert_allclose(d['unit_vectors'], uv, atol=2e-2)
            
            for k, (_, d) in enumerate(c.iter(data=['curvature'])):
                self.assertAlmostEqual(d['curvature'], 1/r, 4)

            # test plane intersections
            p = c.plane_intersects((0, 0, 0), [1, 0, 0])
            np.testing.assert_allclose(p, [[0, r, 0], [0, -r, 0]], rtol=1e-4)
                    
            # run the other way round
            c.invert_parameterization()
            a = a[::-1]
            
            for k, (_, d) in enumerate(c.iter(data=['unit_vectors'])):
                uv = [[-np.cos(a[k]),  np.sin(a[k]), 0],
                      [-np.sin(a[k]), -np.cos(a[k]), 0],
                      [            0,             0, 1]]
                np.testing.assert_allclose(d['unit_vectors'], uv, atol=2e-2)
            
            for k, (_, d) in enumerate(c.iter(data='all')):
                self.assertAlmostEqual(d['curvature'], 1/r, 4)
                f = 0.5 if k == 0 or k == len(a) - 1 else 1
                self.assertAlmostEqual(d['stretching_factor'],
                                       2*np.pi*r*f / (len(a) - 1), 5)
        
        
    def test_expanded_helix(self):
        """ test expanded helix curve """
        for smoothing in (0, 0.01):
            # define the discretized curve
            a, b = 1, 0.1
            t, dt = np.linspace(0, 20, 512, retstep=True)
            ps = np.c_[a*np.cos(t), a*np.sin(t), np.exp(b*t)]
            c = Curve3D(ps, smoothing)
            
            # calculate the tangent vector and other data
            f = b**2 * np.exp(2*b*t)
            arc_len = np.sqrt(a**2 + f) * dt
            
            denom = np.sqrt(a**2 + f)[:, None]
            tangent = np.c_[-a*np.sin(t), a*np.cos(t), b*np.exp(b*t)] / denom
            
            denom = np.sqrt((a**2 + f) * (a**2 + f * (1 + b**2)))
            normal = np.c_[-a**2*np.cos(t) + f*(b*np.sin(t) - np.cos(t)),
                           -a**2*np.sin(t) - f*(b*np.cos(t) + np.sin(t)),
                           a * b**2 * np.exp(b*t)
                           ] / denom[:, None]
                
            denom = np.sqrt(a**2 + f * (1 + b**2))
            binormal = np.c_[b*np.exp(b*t) * (b*np.cos(t) + np.sin(t)),
                             b*np.exp(b*t) * (b*np.sin(t) - np.cos(t)),
                             np.full_like(t, a)] / denom[:, None]
                           
            curvature = (a * np.sqrt(a**2 + f * (1 + b**2)) /
                            (a**2 + f) ** (3/2))
            
            torsion = (b + b**3) * np.exp(b*t) / (a**2 + (1 + b**2)*f)
                           
            self.assertAlmostEqual(c.length, arc_len.sum(), 1)
            self.assertAlmostEqual(c.length, c.stretching_factors.sum())
            
            for k, (_, d) in enumerate(c.iter(data='all')):
                np.testing.assert_allclose(d['tangent'], tangent[k], atol=0.05)
                np.testing.assert_allclose(d['normal'], normal[k], atol=0.05)
                np.testing.assert_allclose(d['binormal'], binormal[k],
                                           atol=0.05)
                np.testing.assert_allclose(d['curvature'], curvature[k],
                                           atol=0.1)
                if 1 < k < len(t) - 2:
                    np.testing.assert_allclose(d['torsion'], torsion[k],
                                               atol=0.1, err_msg='%d' % k )
                f = 0.5 if k == 0 or k == len(t) - 1 else 1
                np.testing.assert_allclose(d['stretching_factor'], f*arc_len[k],
                                           atol=0.01, rtol=0.2)
        
        
    def test_random_curve(self):
        """ generate a random curve """
        ps = np.random.rand(32, 3)
        ps += np.arange(32)[:, None]
        
        c = Curve3D(ps)
        c = c.make_equidistant(count=64)
        c = c.make_smooth()
        
        for _, d in c.iter(data=['unit_vectors']):
            
            u, v, w = d['unit_vectors']
            self.assertAlmostEqual(np.dot(u, v), 0, 1)
            self.assertAlmostEqual(np.dot(v, w), 0, 1)
            self.assertAlmostEqual(np.dot(u, w), 0, 1)
            
            # check scalar triple product:
            triple = np.dot(u, np.cross(v, w))
            self.assertAlmostEqual(triple, 1, 3)

        c_len = c.length
        c2 = c.make_equidistant(count=50)
        self.assertNotEqual(c_len, c2.length)  # check that something changed
        self.assertAlmostEqual(c_len, c2.length, places=2)  # but not too much

        c2 = c.make_smooth(smoothing=10)
        self.assertNotEqual(c_len, c2.length)  # check that something changed
        self.assertAlmostEqual(c_len, c2.length, places=1)  # but not too much

            
    def test_corner_case(self):
        """ test some more complicated cases """
        for smoothing in (0, 0.1):
            ps = np.zeros((4, 3))
            c = Curve3D(ps, smoothing)
            for _, d in c.iter(data='all'):
                self.assertTrue(np.all(np.isnan(d['unit_vectors'])))
                self.assertAlmostEqual(d['stretching_factor'], 0)
                self.assertAlmostEqual(d['arc_length'], 0)
    
            ps = np.zeros((4, 3))
            ps[:] = np.arange(4)[:, None]
            c = Curve3D(ps)
            tangent = np.ones(3) / np.sqrt(3)
            for k, (_, d) in enumerate(c.iter(data='all')):
                np.testing.assert_array_equal(d['tangent'], tangent)
                self.assertTrue(np.all(np.isnan(d['normal'])))
                self.assertTrue(np.all(np.isnan(d['binormal'])))
                self.assertAlmostEqual(d['arc_length'], k * np.sqrt(3))
                f = 0.5 if k == 0 or k == 3 else 1
                self.assertAlmostEqual(d['stretching_factor'], f * np.sqrt(3))



if __name__ == "__main__":
    unittest.main()
