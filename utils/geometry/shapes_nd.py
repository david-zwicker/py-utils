'''
Created on Nov 5, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import itertools

import numpy as np



class Line(object):
    """ represents a line in n dimensions """
    
    
    def __init__(self, origin, direction, mutable=False):
        """ initialize the line with an `origin` point and a `direction` """
        self._mutable = mutable 
        self.origin = asanyarray_flags(origin, np.double,
                                       writeable=self.mutable) 
        self.direction = direction  # normalizes the direction
        
    
    @property
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, value):
        self._direction = np.array(value, np.double)  # make copy
        if self.origin.shape != self._direction.shape:
            raise ValueError('Direction vector must have the same dimension as '
                             'the origin vector')
        self._direction /= np.linalg.norm(self._direction)
        self._direction.flags.writeable = self.mutable    
    
        
    @property
    def mutable(self):
        return self._mutable
    
    @mutable.setter
    def mutable(self, value):
        self._mutable = bool(value)
        self.origin.flags.writeable = self._mutable
        self._direction.flags.writeable = self._mutable

    
    @classmethod
    def from_points(cls, point1, point2, **kwargs):
        point1 = np.asanyarray(point1, np.double)
        point2 = np.asanyarray(point2, np.double)
        return cls(point1, point2 - point1, **kwargs)
        
  
    @property
    def dim(self):
        return len(self.origin)

        
    def __repr__(self):
        return "{cls}(origin={origin}, direction={direction})".format(
                        cls=self.__class__.__name__,
                        origin=self.origin, direction=self.direction)

  
    def contains_point(self, points):
        """ tests whether the points lie on the line """
        p_o = points - self.origin
        return np.isclose(np.abs(np.dot(p_o, self.direction)),
                          np.linalg.norm(p_o, axis=-1))

    
    def project_point(self, points):
        """ projects points onto the line """
        points = np.asanyarray(points, np.double)
        is_1d = (points.ndim == 1)
        
        p_o = points - self.origin
        dist_projection = np.dot(p_o, self.direction)
        res = self.origin + np.outer(dist_projection, self.direction)
        return res[0] if is_1d else res
    
    
    def point_distance(self, points):
        """ calculates the distance of points from the line """
        points = np.asanyarray(points, np.double)
        is_1d = (points.ndim == 1)
        
        p_o = points - self.origin
        dist_projection = np.dot(p_o, self.direction)
        diff_vector = p_o - np.outer(dist_projection, self.direction)
        dist = np.linalg.norm(diff_vector, axis=1)
        return dist[0] if is_1d else dist
    
    
    def distance(self, other):
        """ calculates distance to another line
        
        The math and the notation used here was taken from
        http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment
        """
        if isinstance(other, Line):
            if self.dim != other.dim:
                raise ValueError('Lines must be in space of same dimensions')
            
            p_0 = self.origin
            u = self.direction
            q_0 = other.origin
            v = other.direction
            w_0 = p_0 - q_0
            
            # a = np.dot(u, u)  # == 1 because of normalization
            b = np.dot(u, v)
            # c = np.dot(v, v)  # == 1 because of normalization
            d = np.dot(u, w_0)
            e = np.dot(v, w_0)
            
            denom = 1 - b**2  # == a*c - b**2
            if denom == 0:
                # lines are parallel
                s_c = 0
                t_c = d / b  # Note that b != 0, since b**2 = a*c and a*c != 0
            else:
                # lines are skewed
                s_c = (b*e - d) / denom  # (b*e - c*d) / denom
                t_c = (e - b*d) / denom  # (a*e - b*d) / denom
            
            diff = w_0 + s_c * u - t_c * v
            return np.linalg.norm(diff)
        
        else:
            raise TypeError("Don't know how to calculate distance to type `%s`",
                            other.__class__)



class Segment(object):
    """ represents a segment in n dimensions """
    
    
    def __init__(self, start, end, mutable=False):
        """ initialize the straight segment from point p1 to p2 """
        self._mutable = mutable
        self.start = asanyarray_flags(start, np.double, writeable=self.mutable) 
        self.end = asanyarray_flags(end, np.double, writeable=self.mutable)
        
        # consistency check
        shape = (self.dim,) 
        if self.start.shape != shape or self.end.shape != shape:
            raise ValueError('Both `start` and `end` must be points with the '
                             'same dimension.')
        
        
    @property
    def mutable(self):
        return self._mutable
    
    @mutable.setter
    def mutable(self, value):
        self._mutable = bool(value)
        self.start.flags.writeable = self._mutable
        self.end.flags.writeable = self._mutable
        
        
    @property
    def dim(self):
        return len(self.start)

        
    def __repr__(self):
        return "{cls}(start={start}, end={end})".format(
                    cls=self.__class__.__name__, start=self.start, end=self.end)

  
    @property
    def length(self):
        """ return the length of the segment """
        return np.linalg.norm(self.end - self.start)
    
    
    @property
    def centroid(self):
        """ return the centroid of the segment """
        return 0.5 * (self.start + self.end)
    
    
    @property
    def points(self):
        """ return the two endpoints as a single array """
        return np.vstack((self.start, self.end))
        
    
    @property
    def line(self):
        """ return the line that corresponds to this segment """
        return Line.from_points(self.start, self.end)
  
        
        
class Plane(object):
    """ represents a plane in n dimensions """

    
    def __init__(self, origin, normal, mutable=False):
        """ initialize the plane with an `origin` and a `normal` vector """
        self._mutable = mutable 
        self.origin = asanyarray_flags(origin, np.double,
                                       writeable=self.mutable) 
        self.normal = normal  # normalizes the vector and checks consistency
        
        
    @property
    def normal(self):
        return self._normal
    
    @normal.setter
    def normal(self, value):
        self._normal = np.array(value, np.double)  # make copy
        if self.origin.shape != self._normal.shape:
            raise ValueError('Normal vector (dim=%d) must have the same '
                             'dimension as the origin vector (dim=%d)' %
                             (len(self._normal), len(self.origin)))
        self._normal /= np.linalg.norm(self._normal)
        self._normal.flags.writeable = self.mutable
        
        
    @property
    def mutable(self):
        return self._mutable
    
    @mutable.setter
    def mutable(self, value):
        self._mutable = bool(value)
        self.origin.flags.writeable = self._mutable
        self._normal.flags.writeable = self._mutable
        
    
    @classmethod
    def from_points(cls, points, **kwargs):
        """ estimates a plane from a point cloud """
        points = np.array(points, np.double)  # make copy
        num, dim = points.shape
        if num < dim:
            raise ValueError('At least as many points as dimensions are '
                             'required to define a plane.')

        # center the points
        centroid = np.mean(points, axis=0)
        points = points - centroid[None, :]
        
        # get normal from left singular vector of the smallest singular value
        _, s, v = np.linalg.svd(points, full_matrices=False)
        return cls(centroid, v[np.argmin(s)], **kwargs)
        
    
    @classmethod
    def from_average(cls, planes, **kwargs):
        """ creates a plane by averaging other planes """
        return cls(origin=np.mean([plane.origin for plane in planes], axis=0),
                   normal=np.mean([plane.normal for plane in planes], axis=0),
                   **kwargs)
        
            
    @property
    def dim(self):
        return len(self.origin)
        
        
    def __repr__(self):
        return "{cls}(origin={origin}, normal={normal})".format(
                        cls=self.__class__.__name__,
                        origin=self.origin, normal=self.normal)

    
    def __hash__(self):
        """ custom hash function """
        return hash((tuple(self.origin), tuple(self.normal)))


    def __eq__(self, other):
        """ override the default equality test """
        if isinstance(other, self.__class__):
            return (np.all(self.origin == other.origin) and 
                    np.all(self.normal == other.normal))
        return NotImplemented


    def __ne__(self, other):
        """ overwrite the default non-equality test """
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented
    
    
    def copy(self, origin=None, normal=None):
        """ create a copy of the current plane with the given attributes """
        return self.__class__(origin=self.origin if origin is None else origin,
                              normal=self.normal if normal is None else normal)
    
    
    def distance_point(self, points, oriented=False):
        """ calculates the distance of points to the plane
        If `oriented` is True, the oriented distance is returned, which is
        positive if the point lies in the half space in which the normal points
        """
        p_o = points - self.origin
        distance = np.dot(p_o, self.normal)
        if oriented:
            return distance
        else:
            return np.abs(distance)

    
    def contains_point(self, points):
        """ tests whether the points lie on the plane """
        return np.isclose(self.distance_point(points), 0)

    
    def project_point(self, points):
        """ projects points onto this plane and returns the new coordinates """
        points = np.asanyarray(points, np.double)
        is_1d = (points.ndim == 1)
        
        p_o = points - self.origin
        dist = np.dot(p_o, self.normal)  # oriented distance
        res = points - np.outer(dist, self.normal)
        return res[0] if is_1d else res
    
    
    def flip_normal(self):
        """ returns a plane with the normal flipped """
        return self.copy(normal=-self.normal)


    def translate(self, vector=0, inplace=False):
        """ translates the plane by the `vector` """
        vector = np.asarray(vector)
        if inplace:
            self.origin += vector
            return self
        else:
            return self.copy(origin=self.origin + vector)

    

class Cuboid(object):
    """ class that represents a cuboid in n dimensions """
    
    
    def __init__(self, pos, size, mutable=True):
        """ defines a cuboid from a position of one corner and a vector defining
        its size """
        self._mutable = mutable
        self.pos = asanyarray_flags(pos, writeable=self.mutable)
        self.size = size  # implicitly sets correct shape 
        
        
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        self._size = np.array(value, self.pos.dtype)  # make copy
        if self.pos.shape != self._size.shape:
            raise ValueError('Size vector (dim=%d) must have the same '
                             'dimension as the position vector (dim=%d)' %
                             (len(self._size), len(self.pos)))
        
        # flip Cuboid with negative size
        neg = (self._size < 0)
        self.pos[neg] += self._size[neg]
        self._size = np.abs(self._size)
        self._size.flags.writeable = self.mutable
        
        
    @property
    def mutable(self):
        return self._mutable
    
    @mutable.setter
    def mutable(self, value):
        self._mutable = bool(value)
        self.pos.flags.writeable = self._mutable
        self._size.flags.writeable = self._mutable

        
    @classmethod
    def from_points(cls, p1, p2, **kwargs):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        return cls(p1, p2 - p1, **kwargs)
    
    
    @classmethod
    def from_bounds(cls, bounds, **kwargs):
        bounds = np.asarray(bounds).reshape(-1, 2)
        return cls(bounds[:, 0], bounds[:, 1] - bounds[:, 0], **kwargs)
    
    
    @classmethod
    def from_centerpoint(cls, centerpoint, size, **kwargs):
        centerpoint = np.asarray(centerpoint)
        size = np.asarray(size)
        return cls(centerpoint - size / 2, size, **kwargs)
    
    
    def copy(self):
        return self.__class__(self.pos, self.size)
        
        
    def __repr__(self):
        return "{cls}(pos={pos}, size={size})".format(
                        cls=self.__class__.__name__,
                        pos=self.pos, size=self.size)
            
            
    def __eq__(self, other):
        """ override the default equality test """
        if isinstance(other, self.__class__):
            return (np.all(self.pos == other.pos) and 
                    np.all(self.size == other.size))
        return NotImplemented


    def __ne__(self, other):
        """ overwrite the default non-equality test """
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented
    
    
    @property
    def dim(self):
        return len(self.pos)
            

    @property
    def bounds(self):
        return [(p, p + s) for p, s in zip(self.pos, self.size)]
            
    
    @property
    def corners(self):
        """ return coordinates of two extreme corners defining the cuboid """
        return np.copy(self.pos), self.pos + self.size
    
    @corners.setter
    def corners(self, points):
        """ set coordinates of two extreme corners defining the cuboid """
        points = np.asarray(points)
        if points.shapes != (2, self.dim):
            raise ValueError('Expect two points with correct dimensions.')
        self.pos[:] = points[0, :]
        self.size = points[1, :] - points[0, :]


    @property
    def vertices(self):
        """ return the coordinates of all the corners """
        return list(itertools.product(*self.bounds))


    @property
    def diagonal(self):
        """ returns the length of the diagonal """
        return np.linalg.norm(self.size)
    
    
    @property
    def surface_area(self):
        """ generalization of the surface area of a cuboid in n dimensions. The
        surface area is the volume of the (n-1)-dimensional hypercubes that
        bound the current cuboid:
            n=1, the number of end points (2)
            n=2, the perimeter of the rectangle
            n=3, the surface area of the cuboid
        """
        sides = self.size
        null = (sides == 0)
        null_count = null.sum()
        if null_count == 0:
            return 2 * np.sum(np.product(sides) / sides)
        elif null_count == 1:
            return 2 * np.product(sides[~null])
        else:
            return 0


    @property
    def slices(self):
        return [slice(int(p), int(p + s)) for p, s in zip(self.pos, self.size)]


    @property
    def centroid(self):
        return self.pos + self.size / 2
    
    @centroid.setter
    def centroid(self, center):
        self.pos[:] = np.asanyarray(center) - self.size / 2
    
    
    @property
    def volume(self):
        return np.prod(self.size)
    
    
    def contains_point(self, points):
        """ returns a True when `points` are within the Cuboid """
        points = np.asarray(points)
        if len(points) == 0:
            return points
        
        if points.shape[-1] != self.dim:
            raise ValueError('Last dimension of `points` must agree with '
                             'cuboid dimension %d' % self.dim)
        
        c1, c2 = self.corners
        return np.all(c1 <= points, axis=-1) & np.all(points <= c2, axis=-1)
    

    def translate(self, distance=0, inplace=False):
        """ translates the cuboid by a certain distance in all directions """
        distance = np.asarray(distance)
        if inplace:
            self.pos += distance
            return self
        else:
            return self.__class__(self.pos + distance, self.size)
    
            
    def buffer(self, amount=0, inplace=False):
        """ dilate the cuboid by a certain amount in all directions """
        amount = np.asarray(amount)
        if inplace:
            self.pos -= amount
            self.size += 2*amount
            return self
        else:
            return self.__class__(self.pos - amount, self.size + 2*amount)
    

    def scale(self, factor=1, inplace=False):
        """ scale the cuboid by a certain amount in all directions. The corner
        with the smallest coordinates is used as the center """
        factor = np.asarray(factor)
        if inplace:
            self.pos *= factor
            self.size *= factor
            return self
        else:
            return self.__class__(self.pos * factor, self.size * factor)
        
        
    def extend(self, direction, magnitude=1, inplace=False):
        """ extends the box in a given direction by the given `magnitude` """
        direction = np.asanyarray(direction)
        if direction.shape != self.pos.shape:
            raise ValueError('`direction` must have same dimensions as box')
        
        dir_neg = (direction < 0)
        if inplace: 
            self.size += np.abs(direction) * magnitude
            self.pos[dir_neg] += direction[dir_neg] * magnitude
            return self
        
        else:
            size = self.size + np.abs(direction) * magnitude
            pos = self.pos.copy()
            pos[dir_neg] += direction[dir_neg] * magnitude
            return self.__class__(pos, size)
        
        
    def adjust_side(self, axis, direction, position, inplace=False):
        """ adjust the position of a single side specified by the `axis` and a
        `direction`, which is either `1` or `-1`. The position of the side will
        be set to `position` """
        if inplace:
            if direction == 1:
                self.size[axis] = position - self.pos[axis]
            elif direction == -1:
                self.size[axis] -= position - self.pos[axis]
                self.pos[axis] = position
            else:
                RuntimeError('Direction should be either 1 or -1')
            return self
                
        else:
            pos, size = self.pos.copy(), self.size.copy()
            if direction == 1:
                size[axis] = position - pos[axis]
            elif direction == -1:
                size[axis] -= position - pos[axis]
                pos[axis] = position
            else:
                RuntimeError('Direction should be either 1 or -1')
            return self.__class__(pos, size)
        
        
    def face_plane(self, axis, direction=1):
        """ returns a Plane object that represents a plane through on of the
        sides perpendicular to `axis`. `direction` can be either -1, 0, or 1 and
        determines which of the sides is used. Here, 0 marks the center """
        if not np.isscalar(axis):
            raise TypeError('The `axis` argument must be a scalar that '
                            'determines along which axis the plane is taken.')
        
        origin = np.array(self.centroid)
        origin[axis] += 0.5 * np.sign(direction) * self.size[axis]
        normal = np.zeros(self.dim, np.double)
        normal[axis] = 1 if direction >= 0 else -1
        return Plane(origin, normal)
    


class Cylinder(object):
    """ represents a single cylinder """
    
    
    def __init__(self, center_1, center_2, radius, mutable=False):
        """ defines a cylinder from the two centers of the circles and a
        radius """
        self._mutable = mutable
        self.center_1 = asanyarray_flags(center_1, np.double, self.mutable)
        self.center_2 = asanyarray_flags(center_2, np.double, self.mutable)
        self.radius = float(radius)
        
        # consistency check
        shape = (self.dim,) 
        if self.center_1.shape != shape or self.center_2.shape != shape:
            raise ValueError('Both center points must have the same dimension.')
    
        
    @property
    def mutable(self):
        return self._mutable
    
    @mutable.setter
    def mutable(self, value):
        self._mutable = bool(value)
        self.center_1.flags.writeable = self._mutable
        self.center_2.flags.writeable = self._mutable

    
    @classmethod
    def from_segment(cls, segment, radius):
        """ creates a cylinder from a Segment """
        return cls(segment.start, segment.end, radius)
        
        
    def __repr__(self):
        return '{cls}(center_1={p1}, center_2={p2}, radius={radius})'.format(
                cls=self.__class__, p1=self.center_1, p2=self.center_2,
                radius=self.radius)
        
        
    @property
    def dim(self):
        return len(self.center_1)
        

    @property
    def height(self):
        """ return the height of the cylinder """
        return np.linalg.norm(self.center_2 - self.center_1)


    @property
    def centroid(self):
        """ return the centroid of the segment """
        return 0.5 * (self.center_1 + self.center_2)
    

    def distance_point(self, point):
        """ this does only calculate the distance to the side walls """
        point = np.asanyarray(point, np.double)
        
        # move origin of the coordinate system onto p1
        p_o = point - self.center_1
        direction = self.center_2 - self.center_1
        direction /= self.height  # normalize direction
    
        # get normalized vector from (extended) center axis to point
        s_projection = np.dot(p_o, direction)
        proj_vec = p_o - s_projection * direction
        proj_vec /= np.linalg.norm(proj_vec)
        
        # get the vector to the cylinder surface
        radius_vector = self.radius * proj_vec
        
        # get point along the center axis, restricted to cylinder
        p_axis = np.clip(s_projection, 0, self.height) * direction
        
        # get vector from point to cylinder surface
        dist_centerline = np.linalg.norm(p_axis - p_o)
        if dist_centerline < self.radius:
            return 0
        else:
            dist_surface = np.linalg.norm(p_axis + radius_vector - p_o)
            return dist_surface
        


def asanyarray_flags(data, dtype=None, writeable=True):
    """ turns data into an array and sets the respective flags. A copy is only
    made if necessary """
    try:
        data_writeable = data.flags.writeable
    except AttributeError:
        # `data` did not have the writeable flag => it's not a numpy array  
        result = np.array(data, dtype)
    else:
        if data_writeable != writeable:
            # need to make a copy because the flags differ
            result = np.array(data, dtype)
        else:
            # might have to make a copy to adjust the dtype
            result = np.asanyarray(data, dtype)
            
    # set the flags and return the array
    result.flags.writeable = writeable
    return result
    