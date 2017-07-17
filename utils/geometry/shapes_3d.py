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
            logging.getLogger(__name__).debug('Choose a random up_vector')
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
        
        
    @property
    def up_vector(self):
        return self.basis_v
        
        
    @classmethod
    def from_plane(cls, plane, up_vector=None):
        """ create a coordinate plane from a simple plane """
        if up_vector is None:
            # check whether the given plane has already an up-vector
            try:
                up_vector = plane.basis_v
            except AttributeError:
                pass
            
        return cls(plane.origin, plane.normal, up_vector)
        
        
    def __repr__(self):
        return ("{cls}(origin={origin}, normal={normal}, "
                "up_vector={up_vector})".format(
                        cls=self.__class__.__name__, origin=self.origin,
                        normal=self.normal, up_vector=self.basis_v))

    
    def __getstate__(self):
        """ support for pickling objects """
        # remove private variables, e.g. caches
        return {key: value
                for key, value in self.__dict__.iteritems()
                if not key.startswith('_')}
        
        
    def copy(self, origin=None, normal=None, up_vector=None):
        """ create a copy of the current plane with the given attributes """
        if up_vector is None:
            up_vector = self.basis_v
        return self.__class__(origin=self.origin if origin is None else origin,
                              normal=self.normal if normal is None else normal,
                              up_vector=up_vector)
    
    
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
    
    
    def project_point(self, point, ret_dist=False):
        """ projects a 3d point onto the plane. Returns the coordinates in the
        plane and the distance to the plane. The original point can thus be
        reconstructed using
            self.trans_2d_3d(coords) + self.normal * dist
        """
        coords = self.trans_3d_2d(point)
        if ret_dist:
            dist = np.dot(point - self.trans_2d_3d(coords), self.normal)
            return coords, dist
        else:
            return coords
        
    
    def revert_projection(self, coordinates, distance=0):
        """ reverts a projection operation """
        points = self.trans_2d_3d(coordinates)
        if points.ndim > 1:
            return points + np.outer(distance, self.normal)
        else:
            return points + distance * self.normal
        
        