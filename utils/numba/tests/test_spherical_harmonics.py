'''
Created on Aug 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import math
import random
import unittest

from .. import spherical_harmonics as sh

      
     
class TestSphericalHarmonics(unittest.TestCase):
    """ unit tests for the spherical harmonics """

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel
    
            
    def test_spherical_index(self):
        """ test the conversion of the spherical index """
        # check conversion
        for k in range(20):
            l, m = sh.spherical_index_lm(k)
            self.assertEqual(sh.spherical_index_k(l, m), k)
            
        # check order
        k = 0
        for l in range(4):
            for m in range(-l, l+1):
                self.assertEqual(sh.spherical_index_k(l, m), k)
                k += 1
                
                
    def test_spherical_harmonics_real(self):
        """ test spherical harmonics """
        for l in range(sh.MAX_ORDER_SYM + 1):
            for _ in range(5):
                theta = math.pi * random.random()
                phi = 2 * math.pi * random.random()
                y1 = sh.spherical_harmonic_symmetric_scipy(l, theta)
                y2 = sh.spherical_harmonic_real_scipy(l, 0, theta, phi)
                y3 = sh.spherical_harmonic_symmetric(l, theta)
                self.assertAlmostEqual(y1, y2)
                self.assertAlmostEqual(y1, y3)


    def test_spherical_harmonics_lm(self):
        """ test spherical harmonics """
        for l in range(sh.MAX_ORDER + 1):
            for m in range(-l, l + 1):
                for _ in range(5):
                    k = sh.spherical_index_k(l, m)
                    msg = 'l=%d, m=%d, k=%d' % (l, m, k)
                    theta = math.pi * random.random()
                    phi = 2 * math.pi * random.random()
                    y1 = sh.spherical_harmonic_real_scipy(l, m, theta, phi)
                    y2 = sh.spherical_harmonic_real_scipy_k(k, theta, phi)
                    y3 = sh.spherical_harmonic_real(l, m, theta, phi)
                    y4 = sh.spherical_harmonic_real_k(k, theta, phi)
                    self.assertAlmostEqual(y1, y2, msg=msg)
                    self.assertAlmostEqual(y1, y3, msg=msg)
                    self.assertAlmostEqual(y1, y4, msg=msg)

    

if __name__ == '__main__':
    unittest.main()
