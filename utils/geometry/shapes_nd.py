'''
Created on Nov 5, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np



class Line(object):
    """ represents a line in n dimensions """
    
    def __init__(self, origin, direction):
        self.origin = np.asanyarray(origin, np.double)
        self.direction = direction  # normalizes the normal
        
    
    @property
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, value):
        self._direction = np.asanyarray(value, np.double)
        if self.origin.shape != self._direction.shape:
            raise ValueError('Direction vector must have the same dimension as '
                             'the origin vector')
        self._direction /= np.linalg.norm(self._direction)    
    
    
    @classmethod
    def from_points(cls, point1, point2):
        point1 = np.asanyarray(point1, np.double)
        point2 = np.asanyarray(point2, np.double)
        return cls(point1, point2 - point1)
        
  
    @property
    def dim(self):
        return len(self.origin)

        
    def __repr__(self):
        return "{cls}(origin={origin}, direction={direction})".format(
                        cls=self.__class__.__name__,
                        origin=self.origin, direction=self.direction)

  
    def contains_point(self, points):
        """ tests whether the points lie on the plane """
        p_o = points - self.origin
        return np.isclose(np.abs(np.dot(p_o, self.direction)),
                          np.linalg.norm(p_o, axis=-1))

    
    def project_point(self, points):
        """ projects points onto the line """
        points = np.asanyarray(points, np.double)
        is_1d = (points.ndim == 1)
        
        p_o = points - self.origin
        dist = np.dot(p_o, self.direction)
        res = self.origin + np.outer(dist, self.direction)
        return res[0] if is_1d else res



class Plane(object):
    """ represents a plane in n dimensions """
    
    def __init__(self, origin, normal):
        self.origin = np.asanyarray(origin, np.double)
        self.normal = normal  # normalizes the normal
        assert len(origin) == len(normal)
            
    @property
    def normal(self):
        return self._normal
    
    @normal.setter
    def normal(self, value):
        self._normal = np.asanyarray(value, np.double)
        if self.origin.shape != self._normal.shape:
            raise ValueError('Normal vector (dim=%d) must have the same '
                             'dimension as the origin vector (dim=%d)' %
                             (len(self._normal), len(self.origin)))
        self._normal /= np.linalg.norm(self._normal)
        
    
    @classmethod
    def from_points(cls, points):
        points = np.asanyarray(points, np.double)
        num, dim = points.shape
        if num < dim:
            raise ValueError('At least as many points as dimensions are '
                             'required to define a plane.')

        # center the points
        centroid = np.mean(points, axis=0)
        points = points - centroid[None, :]
        
        # get normal from left singular vector of the smallest singular value
        _, s, v = np.linalg.svd(points, full_matrices=False)
        return cls(centroid, v[np.argmin(s)])
        
            
    @property
    def dim(self):
        return len(self.origin)
        
        
    def __repr__(self):
        return "{cls}(origin={origin}, normal={normal})".format(
                        cls=self.__class__.__name__,
                        origin=self.origin, normal=self.normal)


    def distance_point(self, points):
        """ calculates the distance of points to the plane """
        p_o = points - self.origin
        return np.dot(p_o, self.normal)

    
    def contains_point(self, points):
        """ tests whether the points lie on the plane """
        return np.isclose(self.distance_point(points), 0)

    
    def project_point(self, points):
        """ projects points onto this plane and returns the new coordinates """
        points = np.asanyarray(points, np.double)
        is_1d = (points.ndim == 1)
        
        p_o = points - self.origin
        dist = np.dot(p_o, self.normal)
        res = points - np.outer(dist, self.normal)
        return res[0] if is_1d else res

    

class Cuboid(object):
    """ class that represents a cuboid in n dimensions """
    
    def __init__(self, pos, size):
        self.pos = np.asarray(pos)
        self.size = np.asarray(size)
        assert len(self.pos) == len(self.size)
        
    @classmethod
    def from_points(cls, p1, p2):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        return cls(np.minimum(p1, p2), np.abs(p1 - p2))
    
    @classmethod
    def from_centerpoint(cls, centerpoint, size):
        centerpoint = np.asarray(centerpoint)
        size = np.asarray(size)
        return cls(centerpoint - size/2, size)
    
    def copy(self):
        return self.__class__(self.pos, self.size)
        
    def __repr__(self):
        return "{cls}(pos={pos}, size={size})".format(
                        cls=self.__class__.__name__,
                        pos=self.pos, size=self.size)
            
            
    @property
    def dim(self):
        return len(self.pos)
            
            
    def set_corners(self, p1, p2):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        self.pos = np.minimum(p1, p2)
        self.size = np.abs(p1 - p2)

    @property
    def bounds(self):
        return [(p, p + s) for p, s in zip(self.pos, self.size)]
            
    @property
    def corners(self):
        return self.pos, self.pos + self.size
    @corners.setter
    def corners(self, ps):
        self.set_corners(ps[0], ps[1])

    @property
    def dimension(self):
        return len(self.pos)
        
    @property
    def slices(self):
        return [slice(int(p), int(p + s)) for p, s in zip(self.pos, self.size)]

    @property
    def centroid(self):
        return [p + s/2 for p, s in zip(self.pos, self.size)]
    
    @property
    def volume(self):
        return np.prod(self.size)
    

    def translate(self, distance=0, inplace=True):
        """ translates the cuboid by a certain distance in all directions """
        distance = np.asarray(distance)
        if inplace:
            self.pos += distance
            return self
        else:
            return self.__class__(self.pos + distance, self.size)
    
            
    def buffer(self, amount=0, inplace=True):
        """ dilate the cuboid by a certain amount in all directions """
        amount = np.asarray(amount)
        if inplace:
            self.pos -= amount
            self.size += 2*amount
            return self
        else:
            return self.__class__(self.pos - amount, self.size + 2*amount)
    

    def scale(self, factor=1, inplace=True):
        """ scale the cuboid by a certain amount in all directions """
        factor = np.asarray(factor)
        if inplace:
            self.pos *= factor
            self.size *= factor
            return self
        else:
            return self.__class__(self.pos * factor, self.size * factor)
    

