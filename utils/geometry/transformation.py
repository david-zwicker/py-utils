'''
Created on Jan 23, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np
import yaml

from ..files import open_filename



class AffineTransformation(object):
    """ class that represents an affine coordinate transformation """
    
    
    def __init__(self, matrix, offset=0):
        """ initialize the affine transformation
        
        `matrix` is the linear scaling
        `offset` denotes the constant offset
        """
        self.matrix = np.atleast_2d(matrix)
        self.offset = offset
        self._logger = logging.getLogger(__name__)
    
    
    @property
    def offset(self):
        return self._offset
    
    @offset.setter
    def offset(self, value):
        if np.isscalar(value):
            self._offset = np.full(self.dim_to, value, np.double)
        else:
            self._offset = np.asanyarray(value, np.double)
            if len(self._offset) != self.dim_to:
                raise ValueError('Dimensionality of the offset (%d) must be '
                                 'compatible with the target dimension (%d)' % 
                                 (len(self._offset), self.dim_to))
    
    
    def __eq__(self, other):
        return (np.allclose(self.matrix, other.matrix) and
                np.allclose(self.offset, other.offset))
            
    
    def __getstate__(self):
        return {'matrix': self.matrix, 'offset': self.offset}
    
    
    def __setstate__(self, state):
        self.matrix = state['matrix']
        self.offset = state['offset']

    
    @property
    def dim_to(self):
        return self.matrix.shape[0]
    
    
    @property
    def dim_from(self):
        return self.matrix.shape[1]
    
        
    @classmethod
    def from_scaling(cls, scale, offset=0):
        """ initialize the coordinate transform """
        return cls(np.diag(scale), offset)
    
            
    @classmethod
    def from_file(cls, filename, dimension=3):
        """ read the definition of the affine coordinate transformation from a 
        file. The file given by `filename` can either be a string or an already
        opened file handle.
        
        `dimension` specifies the default dimension if it cannot be determined
            from the file
        """
        with open_filename(filename, "r") as fp:
            # read file defining coordinate transform
            data = yaml.load(fp)
        
        # read parameters directly
        matrix = data.get('transformation_matrix', None)
        offset = data.get('offset_vector', 0)
        
        if matrix is None:
            # read dimension
            try:
                dim = int(data['dimension'])
            except KeyError:
                if matrix is not None:
                    dim = len(matrix)
                else:
                    dim = dimension
                    
            matrix = np.eye(dim)

        obj = cls(matrix, offset)
        obj._logger.debug('Read %s from file', obj)
        return obj
    
    
    def write_to_yaml(self, file_handle):
        """ save the definition of the affine transformation to a file """
        data = {'offset_vector': self.offset.tolist(),
                'transformation_matrix': self.matrix.tolist()}
        file_handle.write(yaml.dump(data, default_flow_style=None))

        self._logger.debug('Wrote %s to file', self)

     
    def __repr__(self):
        """ return a string representing the object """ 
        return '{name}(matrix={matrix}, offset={offset})'.format(
                    name=self.__class__.__name__, matrix=self.matrix,
                    offset=self.offset)
            
            
    @property
    def is_identity(self):
        """ returns True iff the transformation is the identity """
        return (self.dim_to == self.dim_from and
                np.all(self.offset == 0) and
                np.all(self.matrix == np.eye(self.dim_to)))
    
    
    @property
    def is_scaling(self):
        """ returns True if transformation is one-to-one and a pure scaling and
        some offset, i.e., if there is no shear or rotation. """
        return (self.dim_to == self.dim_from and
                np.allclose(self.matrix, np.diag(np.diag(self.matrix))))
    
    
    def inverse(self, warn=True):
        """ returns the inverse transform """
        if self.dim_to == self.dim_from:
            # create the inverse transformation
            matrix_inv = np.linalg.inv(self.matrix)
            offset_inv = -np.dot(matrix_inv, self.offset)
            trans = self.__class__(matrix_inv, offset_inv)
    
        else:  # self.dim_to != self.dim_from
            # create the inverse transformation using a pseudo-inverse
            if warn:
                self._logger.warn('Inverse transform is constructed using '
                                  'pseudo-inverse.')
            matrix_inv = np.linalg.pinv(self.matrix)
            offset_inv = -np.dot(matrix_inv, self.offset)
            trans = self.__class__(matrix_inv, offset_inv)
            
        return trans
    
    
    def apply(self, coords):
        """ apply the coordinate transformation
        If multiple coordinates are supplied, the last axis is assumed to be the
        one containing the actual coordinates.
        """
        return self.offset + np.einsum('ij,...j->...i', self.matrix, coords)
    
    
    def apply_inverse(self, coords):
        """ apply the inverse of the coordinate transformation """ 
        if self.dim_to == self.dim_from:
            return np.linalg.solve(self.matrix, coords - self.offset)
        else:
            return np.linalg.lstsq(self.matrix, coords - self.offset)[0]
        
        
    def project(self, coords):
        """ applies the transformation and the inverse successively. This is a 
        projection operation if the defined transformation is toward a lower
        dimensions. In any other case, this does not do anything. """
        return self.apply_inverse(self.apply(coords))
    
    
    def __call__(self, coords):
        """ apply the coordinate transformation
        If multiple coordinates are supplied, the last axis is assumed to be the
        one containing the actual coordinates.
        """
        return self.apply(coords)
        
        