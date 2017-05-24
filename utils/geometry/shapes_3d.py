'''
Created on May 24, 2017

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np

from . import shapes_nd, transformation
from ..data_structures.cache import cached_property



class CoordinatePlane(shapes_nd.Plane):
    """ represents a 2d plane embedded in 3d space equipped with a local
    coordinate system """
 
    def __init__(self, origin, normal, up_vector=None):
        """ initialize the plane using a position vector to the origin, a normal
        vector defining the plane orientation and a basis vector denoting the
        up-direction """
        # this also normalizes the normal vector
        super(CoordinatePlane, self).__init__(origin, normal)
        
        if self.dim != 3:
            raise NotImplementedError('CoordiantePlanes can only be embedded '
                                      'in 3d space, yet.')
        
        if up_vector is None:
            # choose random up_vector until its sufficiently perpendicular
            logging.debug('Choose a random up_vector')
            while True:
                up_vector = np.random.rand(self.dim)
                up_vector_norm = np.linalg.norm(up_vector)
                if np.dot(up_vector, self.normal) < 0.99 * up_vector_norm:
                    break
        
        else:
            up_vector = np.asanyarray(up_vector)
            
        # project `up_vector` onto plane
        up_vector_proj = np.dot(up_vector, self.normal) * self.normal
        self.basis_v = up_vector - up_vector_proj
        self.basis_v /= np.linalg.norm(self.basis_v)  # normalize

        # check consistency
        if not np.all(np.isfinite(self.basis_v)):
            raise RuntimeError('Could not determine a consistent basis using '
                               'the normal vector %s and the up vector %s' % 
                               (self.normal, up_vector))
        
        # calculate an orthogonal vector 
        self.basis_u = -np.cross(self.normal, self.basis_v)
        
        
    @classmethod
    def from_plane(cls, plane, up_vector=None):
        """ create a coordinate plane from a simple plane """
        return cls(plane.origin, plane.normal, up_vector)
        
    
    def __getstate__(self):
        """ support for pickling objects """
        # remove private variables, e.g. caches
        return {key: value
                for key, value in self.__dict__.iteritems()
                if not key.startswith('_')}
        
    
    @cached_property()
    def trans_2d_3d(self):
        """ gives the coordinate transformation from the local 2d space to the
        global 3d coordinate system """
        matrix = np.c_[self.basis_u, self.basis_v]
        return transformation.AffineTransformation(matrix, self.origin)


    @cached_property()
    def trans_3d_2d(self):
        """ gives the coordinate transformation from 3d space to the local 2d
        coordinate system """
        return self.trans_2d_3d.inverse(warn=False)
        
        