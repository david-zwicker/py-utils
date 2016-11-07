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
        
        for k, (p, d) in enumerate(c.iter(data=['tangent', 'arc_length'])):
            np.testing.assert_array_equal(p, ps[k])
            np.testing.assert_array_equal(d['tangent'], [0, 0, 1])
            self.assertAlmostEqual(d['arc_length'], 1)
            
            
    def test_circle(self):
        """ test circle """
        r = 2
        a = np.linspace(0, 2 * np.pi, 256)
        ps = np.c_[r * np.sin(a), r * np.cos(a), np.zeros_like(a)]
        c = Curve3D(ps)
        
        self.assertAlmostEqual(c.length, 2 * np.pi * r, places=2)
        np.testing.assert_array_equal(list(c), ps)
        
        # check the individual vectors
        data = ['tangent', 'normal', 'binormal', 'arc_length']
        for k, (p, d) in enumerate(c.iter(data=data)):
            np.testing.assert_array_equal(p, ps[k])
            np.testing.assert_allclose(d['tangent'],
                                       [np.cos(a[k]), -np.sin(a[k]), 0],
                                       atol=2e-2)
            np.testing.assert_allclose(d['normal'],
                                       [-np.sin(a[k]), -np.cos(a[k]), 0],
                                       atol=2e-2)
            np.testing.assert_allclose(d['binormal'], [0, 0, -1])
            self.assertAlmostEqual(d['arc_length'], 2*np.pi*r / (len(a) - 1), 5)
        
        # check the unit_vector system
        for k, (p, d) in enumerate(c.iter(data=['unit_vectors'])):
            np.testing.assert_array_equal(p, ps[k])
            uv = [[ np.cos(a[k]), -np.sin(a[k]),  0],
                  [-np.sin(a[k]), -np.cos(a[k]),  0],
                  [            0,             0, -1]]
            np.testing.assert_allclose(d['unit_vectors'], uv, atol=2e-2)
        
        for k, (_, d) in enumerate(c.iter(data=['curvature'])):
            if 0 < k < len(a) - 1:
                self.assertAlmostEqual(d['curvature'], 1/r, 4)
                
        # run the other way round
        c.invert_parameterization()
        a = a[::-1]
        
        for k, (_, d) in enumerate(c.iter(data=['unit_vectors'])):
            uv = [[-np.cos(a[k]),  np.sin(a[k]), 0],
                  [-np.sin(a[k]), -np.cos(a[k]), 0],
                  [            0,             0, 1]]
            np.testing.assert_allclose(d['unit_vectors'], uv, atol=2e-2)
        
        for k, (_, d) in enumerate(c.iter(data=['curvature', 'arc_length'])):
            if 0 < k < len(a) - 1:
                self.assertAlmostEqual(d['curvature'], 1/r, 4)
            self.assertAlmostEqual(d['arc_length'], 2*np.pi*r / (len(a) - 1), 5)
        
        
    def test_expanded_helix(self):
        """ test expanded helix curve """
        # define the discretized curve
        a, b = 1, 0.1
        t, dt = np.linspace(0, 20, 512, retstep=True)
        ps = np.c_[a*np.cos(t), a*np.sin(t), np.exp(b*t)]
        c = Curve3D(ps)
        
        # calculate the tangent vector and other data
        arc_len = np.sqrt(a**2 + b**2 * np.exp(2*b*t)) * dt
        
        denom = np.sqrt(a**2 + b**2 * np.exp(2*b*t))[:, None]
        tangent = np.c_[-a*np.sin(t), a*np.cos(t), b*np.exp(b*t)] / denom
        
        denom = np.sqrt((a**2 + b**2 * np.exp(2*b*t)) *
                        (a**2 + b**2 * (1 + b**2) * np.exp(2*b*t)))
        normal = \
            np.c_[
                -a**2*np.cos(t) + b**2*np.exp(2*b*t)*(b*np.sin(t) - np.cos(t)),
                -a**2*np.sin(t) - b**2*np.exp(2*b*t)*(b*np.cos(t) + np.sin(t)),
                a * b**2 * np.exp(b*t)
            ] / denom[:, None]
            
        denom = np.sqrt(a**2 + b**2 * (1 + b**2) * np.exp(2*b*t))
        binormal = np.c_[b*np.exp(b*t) * (b*np.cos(t) + np.sin(t)),
                         b*np.exp(b*t) * (b*np.sin(t) - np.cos(t)),
                         np.full_like(t, a)] / denom[:, None]
                       
        curvature = (a * np.sqrt(a**2 + b**2 * (1 + b**2) * np.exp(2 * b * t)) /
                        (a**2 + b**2 * np.exp(2. * b * t)) ** (3/2))
        curvature[0] = 0
        curvature[-1] = 0
                       
        self.assertAlmostEqual(c.length, arc_len.sum(), 1)
        
        data = ['tangent', 'normal', 'binormal', 'arc_length', 'curvature']
        for k, (_, d) in enumerate(c.iter(data=data)):
            np.testing.assert_allclose(d['tangent'], tangent[k], atol=0.05)
            np.testing.assert_allclose(d['normal'], normal[k], atol=0.05)
            np.testing.assert_allclose(d['binormal'], binormal[k], atol=0.05)
            np.testing.assert_allclose(d['arc_length'], arc_len[k], atol=0.01)
            np.testing.assert_allclose(d['curvature'], curvature[k], atol=0.1)
        
        
    def test_random_curve(self):
        """ generate a random curve """
        ps = np.random.rand(8, 3)
        ps += np.arange(8)[:, None]
        
        c = Curve3D(ps)
        c.make_equidistant(count=32)
        c.smooth()
        
        for _, d in c.iter(data=['unit_vectors']):
            
            a, b, c = d['unit_vectors']
            self.assertAlmostEqual(np.dot(a, b), 0, 1)
            self.assertAlmostEqual(np.dot(b, c), 0, 1)
            self.assertAlmostEqual(np.dot(a, c), 0, 1)
            
            # check scalar triple product:
            triple = np.dot(a, np.cross(b, c))
            self.assertAlmostEqual(triple, 1, 3)
            
            
    def test_corner_case(self):
        """ test some more complicated cases """
        ps = np.zeros((4, 3))
        c = Curve3D(ps)
        for _, d in c.iter(data=['unit_vectors', 'arc_length']):
            self.assertTrue(np.all(np.isnan(d['unit_vectors'])))
            self.assertAlmostEqual(d['arc_length'], 0)

        ps = np.zeros((4, 3))
        ps[:] = np.arange(4)[:, None]
        c = Curve3D(ps)
        tangent = np.ones(3) / np.sqrt(3)
        data = ['tangent', 'normal', 'binormal', 'arc_length']
        for _, d in c.iter(data=data):
            np.testing.assert_array_equal(d['tangent'], tangent)
            self.assertTrue(np.all(np.isnan(d['normal'])))
            self.assertTrue(np.all(np.isnan(d['binormal'])))
            self.assertAlmostEqual(d['arc_length'], np.sqrt(3))



if __name__ == "__main__":
    unittest.main()
