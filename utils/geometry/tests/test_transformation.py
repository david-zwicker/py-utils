'''
Created on Aug 11, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np
import six

from utils.geometry.transformation import AffineTransformation
from utils.testing import TestBase



class TestTransformations(TestBase):

    
    def _random_transform(self, pure_scaling=False):
        """ generates a random transform """
        dim = np.random.randint(4, 10)
        offset = np.random.random(dim)
        if pure_scaling:
            scale = np.random.random(dim)
            trans = AffineTransformation.from_scaling(scale, offset)
        else:
            matrix = np.random.random((dim, dim))
            trans = AffineTransformation(matrix, offset)
        return trans
    
    
    def test_simple(self):
        """ test simple properties of transformations """
        trans_scaling = self._random_transform(pure_scaling=True)
        trans_general = self._random_transform(pure_scaling=False)
        
        self.assertTrue(trans_scaling.is_scaling)
        self.assertFalse(trans_scaling.is_identity)
        self.assertFalse(trans_general.is_scaling)
        self.assertFalse(trans_general.is_identity)
    
    
    def test_inverse(self):
        """ tests the inverse of the transform """
        trans = self._random_transform()
        c1 = np.random.random(trans.dim_from)
        c2 = trans(c1)
        
        self.assertAllClose(c1, trans.apply_inverse(c2))

        trans_inv = trans.inverse()
        self.assertEqual(trans.dim_to, trans_inv.dim_from)
        self.assertEqual(trans.dim_from, trans_inv.dim_to)
        self.assertAllClose(c1, trans_inv(c2))
        
        trans_inv2 = trans_inv.inverse()
        self.assertEqual(trans, trans_inv2)
        
        p = np.random.random(trans.dim_from)
        self.assertAllClose(p, trans.projection(p))
    
    
    def test_pseudo_inverse(self):
        """ tests the inverse of the transform """
        # define a 3d plane 
        origin = np.random.randn(3)
        normal = np.random.randn(3)
        normal /= np.linalg.norm(normal)

        # define basis vectors in this plane
        basis1 = np.random.randn(3)
        basis1 -= np.dot(basis1, normal)
        basis1 /= np.linalg.norm(basis1)        
        basis2 = np.cross(basis1, normal)
        
        # define the projection from the 3d space to the 2d subspace
        matrix = np.c_[basis1, basis2]
        trans = AffineTransformation(matrix, origin)
        
        np.testing.assert_allclose(trans(np.zeros(2)), origin)
        
        trans_inv = trans.inverse(warn=False)
        trans_inv2 = trans_inv.inverse(warn=False)
        
        for _ in range(3):
            v2 = np.random.randn(2)  # vector in plane
            v3 = trans(v2)  # vector in space
            np.testing.assert_allclose(trans_inv(v3), v2)
            np.testing.assert_allclose(trans.apply_inverse(v3), v2)
    
        trans_inv3 = trans_inv2.inverse(warn=False)
        self.assertEqual(trans_inv3, trans_inv)

        trans_inv4 = trans_inv3.inverse(warn=False)
        self.assertEqual(trans_inv4, trans_inv2)
            
            
    def test_save_transformation(self):
        """ test saving a transformation to a file """
        trans = self._random_transform()
        self.assertIsInstance(repr(trans), six.string_types)
        
        # write transformation to string buffer        
        string_buffer = six.StringIO()
        trans.write_to_yaml(string_buffer)
        string_buffer.seek(0)
        
        # read the data back
        loaded = AffineTransformation.from_file(string_buffer)
        self.assertEqual(trans, loaded)
        
        
    def test_pickling(self):
        """ test pickling of transformations """
        import pickle
        trans = self._random_transform()        
        trans2 = pickle.loads(pickle.dumps(trans))
        self.assertEqual(trans, trans2)



if __name__ == "__main__":
    unittest.main()
    