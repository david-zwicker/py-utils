'''
Created on Nov 8, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from .. import shapes_2d



class TestShapes2D(unittest.TestCase):

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def _check_edges(self, edges):
        edges = np.r_[edges, edges[0]]  # periodic boundary
        self.assertEqual(np.count_nonzero(np.diff(edges) < 0), 1)
    

    def test_iter_problematic_edges(self):
        """ test the helper function """
        def func(l):
            return list(shapes_2d._iter_problematic_edges(l))
        
        self.assertEqual(func([]), [])
        self.assertEqual(func([0]), [])
        self.assertEqual(func([0, 1]), [])
        self.assertEqual(func(np.arange(10)), [])
        
        self.assertEqual(func([0, 1, 0, 3]), [(1, 3)])
        self.assertEqual(func([0, 1, 3, 1]), [(2, 4)])
        self.assertEqual(func([0, 1, 3, 0]), [])

        self.assertEqual(func([0, 1, 0, 0, 4]), [(1, 4)])
        self.assertEqual(func([0, 2, 0, 1, 4]), [(1, 4)])
        self.assertEqual(func([0, 1, 0, 3, 0, 4]), [(1, 3), (3, 5)])
        

    def test_register_polygons(self):
        """ test the polygon registration with constructed polygons """
        # constructed case
        p1 = np.array([(0, 0), (0, 1), (1, 1), (1, 0)])
        p2 = np.array([(0.3, 1), (1, 1.4), (0.3, 0.6), (1, 0), (0.1, 0)])
        p2 += np.array([0.1, 0.1])
        
        e12, e21 = shapes_2d.register_polygons(p1, p2)
        
        self._check_edges(e12)
        self._check_edges(e21)


    def test_register_polygons_random(self):
        """ test the polygon registration with random polygons """
        def random_poly():
            """ helper function creating a random polygon """
            num = np.random.randint(5, 10)
            a, da = np.linspace(0, 2*np.pi, num, endpoint=False, retstep=True)
            a += np.random.rand(num) * da
            a += 2*np.pi*np.random.rand()  # random phase
            
            r = 0.5 + np.random.rand(num)
            
            return np.c_[r*np.sin(a), r*np.cos(a)]
        
        for _ in range(10):
            p1 = random_poly()
            p2 = random_poly()
    
            e12, e21 = shapes_2d.register_polygons(p1, p2)
            
            self._check_edges(e12)
            self._check_edges(e21)



if __name__ == "__main__":
    unittest.main()
