'''
Created on Nov 4, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

import warnings

import numpy as np
from scipy import interpolate

from six.moves import zip

from utils.data_structures.cache import cached_property



class Curve3D(object):
    ''' represents a curve in 3d space '''

    mutable = False  # determines whether the support points can be changed 


    def __init__(self, points, smoothing_distance=0):
        ''' the curve is given by a collection of linear segments
        
        `points` are the support points defining the curve
        `smoothing_distance` gives the length scale over which calculated
            quantities, like tangent vectors, are smoothed. The default value
            of zero means no smoothing.
        '''
        self.points = points  # implicitly convert to numpy array
        self.smoothing_distance = smoothing_distance
        if self.points.size > 0 and self.points.shape[1] != 3:
            raise ValueError('points must be a nx3 array.')


    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, points):
        """ changes the points that define the curve """
        self._points = np.atleast_2d(points)
        self._points.flags.writeable = self.mutable
        
        if self._points.ndim != 2:
            raise ValueError('Coordinates must be a 2d array')
        if self._points.size > 0 and self._points.shape[-1] != 3:
            raise ValueError('Coordinates must be 3-dimensional')
        self.clear_cache()
        
        
    def clear_cache(self):
        """ clear the cache of curve attributes. The cache is also cleared
        automatically, when the points are set using the `points` attribute.
        Note however, that modifying the `points` attribute in-place does not
        reset the cache! Consequently, `clear_cache` has to be called manually
        after any of the following or similar modifications:
            line.points[0] = 1 
            line.points[1:-1] += 1
        However, these modifications are anyhow only possible if the `mutable`
        is set to True.
        """
        self._cache_methods = {}
        
        
    def copy(self):
        """ return a copy of this class """
        return self.__class__(self.points.copy(), self.smoothing_distance)
    

    @classmethod
    def from_file(cls, filename):
        """ read a polyline from a file. The format is chosen automatically
        based on the file extension """
        if filename.endswith('.xyz'):
            return cls.from_file_xyz(filename)
        else:
            raise ValueError('Do not know how to read file `%s`' % filename)

    
    @classmethod
    def from_file_xyz(cls, filename):
        """ read polyline from a xyz file """
        with open(filename, "r") as fp:
            num_p = int(fp.readline())  # read number of points
            fp.readline()  # skip header
        
            # read the data from the file    
            data = np.loadtxt(fp, dtype=np.double, delimiter=' ',
                              usecols=(1, 2, 3), ndmin=2)
        
        if data.shape != (num_p, 3):
            raise RuntimeError('The shape %s of the data read from the file '
                               '`%s` does not match the expectation (%d, 3).' %
                               (data.shape, filename, num_p))
            
        return cls(data)

    
    @property
    def smoothing_distance(self):
        return self._smoothing_distance
    
    @smoothing_distance.setter
    def smoothing_distance(self, smoothing_distance):
        self._smoothing_distance = smoothing_distance
        # clear cache
        self._cache_methods = {}
        
    
    @cached_property()
    def length(self):
        """ returns the length of the curve """
        return np.linalg.norm(self.points[:-1] - self.points[1:], axis=1).sum()
    
    
    @property
    def num_points(self):
        return len(self.points)
    
    
    @property
    def start(self):
        return self.points[0]
    
    
    @property
    def end(self):
        return self.points[-1]
    
    
    @cached_property()
    def _smoothing_kernel(self):
        """ creates the Gaussian smoothing kernel associated the current line.
        The weights for the different points are based on their distance along
        the curve.
        """
        # TODO: this smoothing kernel weighs each point equally. However, we
        # might want to weigh points according to the local stretching factor
        
        sigma = self.smoothing_distance
        # get a Gaussian kernel based on the arc length `s`
        s = self.arc_lengths
        kernel = np.exp(-(s[:, None] - s[None, :])**2 / (2*sigma**2))
        # normalize the kernel
        kernel /= np.sum(kernel, axis=0, keepdims=True)
        return kernel
        
    
    def _smooth_variable(self, arr):
        """ uses Gaussian smoothing on the supplied array `arr`. The sigma of
        the Gauss filter is given by self.smoothing_distance 
        """ 
        if self.smoothing_distance > 0:
            return np.dot(self._smoothing_kernel, arr)
        else:
            return arr
        

    @cached_property()
    def tangents(self):
        """ return the tangent vector at each support point """
        tangents = np.gradient(self.points, axis=0)
        tangents = self._smooth_variable(tangents)

        return _normalize_vectors(tangents)
    
    
    @cached_property()
    def normals(self):
        """ return the normal vector at each support point """
        normals = np.gradient(self.tangents, axis=0)
        if np.all(normals == 0):
            # the line seems to be a simple straight line
            # => pick a normal direction based on the tangent
            tangent = self.tangents[0]
            normal = np.zeros(3)
            # find axis direction with smallest component
            normal[np.argmin(np.abs(tangent))] = 1
            # make it orthogonal to tangent
            normal -= np.dot(tangent, normal) * tangent
            # use the normal with bigger angle (smaller dot product)
            normals = np.repeat([normal], self.num_points, axis=0)
        return _normalize_vectors(normals)


    @cached_property()
    def binormals(self):
        """ return the binormal vector at each support point """
        binormals = np.cross(self.tangents, self.normals)
        return _normalize_vectors(binormals)
    

    @cached_property()
    def arc_lengths(self):
        """ return the arc length up to each support point """
        ds = np.linalg.norm(self.points[:-1] - self.points[1:], axis=1)
        return np.r_[0, np.cumsum(ds)]    

    
    @cached_property()
    def stretching_factors(self):
        """ return the stretching factor at each support point. The stretching
        factor is the discrete version of ds/dt, where `s` is the arc length and 
        t is the actual parameterization. Consequently, a stretching factor
        of 1 indicates an arc-length parameterization of the curve.
        """
        # calculate half the distance between points
        dist2 = 0.5 * np.linalg.norm(self.points[:-1] - self.points[1:], axis=1)
        factors = np.empty(self.num_points)
        factors[0] = dist2[0]
        factors[1:-1] = dist2[:-1] + dist2[1:]
        factors[-1] = dist2[-1]
        return factors
    
    
    @cached_property()
    def curvatures(self):
        """ return the curvature at each support point """
        if len(self.points) < 3:
            # handle the trivial case
            return np.zeros(len(self.points))
        
        # get the two adjacent vectors to each point and their length
        v1 = self.points[1:-1] - self.points[ :-2]
        v2 = self.points[2:  ] - self.points[1:-1] 
        n1 = np.linalg.norm(v1, axis=-1)
        n2 = np.linalg.norm(v2, axis=-1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # determine angle between successive vectors
            cos_a = np.einsum('ij,ij->i', v1, v2) / (n1 * n2)
            # correct for the local stretching, since we don't enforce
            # arc-length parameterization
            curvatures = np.arccos(cos_a) * 2 / (n1 + n2)
            
        # the curvature at the end points is not well-defined. We here just
        # repeat the values of the adjacent sites  
        curvatures = np.r_[curvatures[0], curvatures, curvatures[-1]]
        return self._smooth_variable(curvatures)
        
        
    @cached_property()
    def torsions(self):
        """ return the torsion at each support point """
        if len(self.points) < 3:
            # handle the trivial case
            return np.zeros(len(self.points))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            dbinormal_dt = np.gradient(self.binormals, axis=0)
            torsions = -np.einsum('ij,ij->i', self.normals, dbinormal_dt)
            torsions /= self.stretching_factors
            
        return self._smooth_variable(torsions)

        
    def iter(self, data=None):
        """ iterates over the points and returns their coordinates
        
        `data` lists extra quantities that are returned for each point. Possible
            values include ('tangent', 'normal', 'binormal', 'unit_vectors',
            'curvature', 'arc_length', 'stretching_factor')
            
        Note that the tangent and normal are calculated from discretized
        derivatives and are thus not necessarily exactly orthogonal. However,
        the binormal vector is guaranteed to be orthogonal to the other two.
        
        Definitions:
            If s(t) is the continuous version of the curve, then
            T(t) = norm(\partial s / \partial_t) is the tangent vector
            N(t) = norm(\partial T / \partial_t) is the normal vector
            B(t) = T(t) \cross N(t) is the binormal vector
            
        The three vectors span the (local) Frenet frame
        """
        if data is None:
            # only return the points, not any extra data
            for p in self.points:
                yield p
        
        else:
            # return extra data
            if data == 'all':
                data = {'tangent', 'normal', 'binormal', 'unit_vectors',
                        'curvature', 'torsion', 'arc_length',
                        'stretching_factor'}
            else:
                data = set(data)

            # calculate requested data
            calculated = {}
            if 'tangent' in data:
                calculated['tangent'] = self.tangents
            if 'normal' in data:
                calculated['normal'] = self.normals
            if 'binormal' in data:
                calculated['binormal'] = self.binormals
            if 'unit_vectors' in data:
                calculated['unit_vectors'] = np.hstack((
                                                self.tangents[:, None, :],
                                                self.normals[:, None, :],
                                                self.binormals[:, None, :]))
                
            if 'stretching_factor' in data:
                calculated['stretching_factor'] = self.stretching_factors
            if 'arc_length' in data:
                calculated['arc_length'] = self.arc_lengths
            if 'curvature' in data:
                calculated['curvature'] = self.curvatures
            if 'torsion' in data:
                calculated['torsion'] = self.torsions

            # return the requested data
            for n, p in enumerate(self.points):
                yield p, {k: calculated[k][n] for k in data}
                
        
    def __iter__(self):
        return self.iter()
    
    
    def invert_parameterization(self):
        """ inverts the parameterization """
        self.points = self.points[::-1]  # clear the cache implicitly 
    
    
    def get_points(self, arc_lengths, interpolation='linear',
                   extrapolate=False):
        """ returns the coordinates of a point at the position specified by
        `arc_length`
        
        `interpolation` determines the used interpolation
        `extrapolate` flag determining whether extrapolation is used to get
            points outside this curve. This only works for `nearest` and
            `linear` interpolation.
        """
        if extrapolate:
            # this requires scipy version 0.17 
            fill_value = 'extrapolate'
        else:
            fill_value = None
        
        interpolator = interpolate.interp1d(
                self.arc_lengths, self.points, kind=interpolation, axis=0,
                copy=False, fill_value=fill_value, assume_sorted=True)
        return interpolator(arc_lengths)


    def plane_intersects(self, origin, normal):
        """ calculates the points where the curve intersects the plane given by
        a position vector `origin` and a normal vector `normal` """
        result = []
        for p1, p2 in zip(self.points, self.points[1:]):
            dp = p2 - p1
            s = np.dot(origin - p1, normal) / np.dot(dp, normal)
            if 0 <= s <= 1:
                result.append(p1 + s * dp)
                
        return np.atleast_2d(result) 

        
    def make_equidistant(self, spacing=None, count=None):
        """ returns a copy of the current curve in which points have been
        chosen equidistantly.
        
        `spacing` gives the approximate spacing between support points
        `count` gives the approximate number of support points
        
        Only either one of these parameters may be supplied. If no parameter is
        given, the number of points is not modified, but they are distributed
        evenly.
        """
        if spacing is not None:
            if count is not None:
                raise ValueError('`spacing` and `count` cannot be specified '
                                 'together') 
            
            # walk along and pick points with given spacing
            if self.length < spacing:
                return
            
            dx = self.length / np.round(self.length / spacing)
            dist = 0
            result = [self.points[0]]
            for p1, p2 in zip(self.points[:-1], self.points[1:]):
                # determine the distance between the last two points 
                dp = np.linalg.norm(p2 - p1)
                # add points to the result list
                while dist + dp > dx:
                    p1 = p1 + (dx - dist)/dp*(p2 - p1)
                    result.append(p1.copy())
                    dp = np.linalg.norm(p2 - p1)
                    dist = 0
                
                # add the remaining distance 
                dist += dp
            
            # add the last point if necessary
            if dist > 1e-8:
                result.append(self.points[-1])
                
        else:
            if count is None:
                count = len(self.points)
                
            # get arc length to all support points
            ps = self.points
            s = np.cumsum(np.linalg.norm(ps[:-1] - ps[1:], axis=1))
            s = np.insert(s, 0, 0)  # prepend element for first point
            # divide arc length equidistantly
            sp = np.linspace(s[0], s[-1], count)
            # interpolate points: TODO: use scipy.interpolation
            result = np.c_[np.interp(sp, s, ps[:, 0]),
                           np.interp(sp, s, ps[:, 1]),
                           np.interp(sp, s, ps[:, 2])]

        # create the new curve            
        return self.__class__(result,
                              smoothing_distance=self.smoothing_distance)
        
    
    def make_smooth(self, smoothing=10, degree=3, derivative=0,
                    num_points=None):
        """ smoothes the curve by interpolating the points
        
        `smoothing` determines the smoothness of the curve.  This value can be
            used to control the trade-off between closeness and smoothness of
            fit. Larger values means more smoothing while smaller values
            indicate less smoothing. The resulting, smoothed yi fulfill
                sum((y - yi)**2, axis=0) <= smoothing*len(points)
        `degree` determines the degree of the splines used
        `derivative` determines the order of the derivative
        `num_points` determines how many support points are generated. If this
            value is None, len(points) are used.
        """
        if num_points is None:
            num_points = len(self.points)
        
        u = np.linspace(0, 1, num_points)
        try:
            # do spline fitting to smooth the line
            tck, _ = interpolate.splprep(np.transpose(self.points), u=u,
                                         k=degree, s=smoothing*len(self.points))
        except ValueError:
            # spline fitting did not work
            warnings.warn('Spline fitting to smooth the curve failed.')
            # at least make the curve equidistant
            line = self.make_equidistant(count=num_points)
                
        else:
            # interpolate the line
            points = interpolate.splev(u, tck, der=derivative)

            # create the new curve            
            line = self.__class__(np.transpose(points),
                                  smoothing_distance=self.smoothing_distance)
            
        return line
            
    
    def write_to_file(self, filename, **kwargs):
        """ write the polyline to a file. The format is chosen automatically
        based on the file extension """
        if filename.endswith('.vtk'):
            self.write_to_vtk(filename, **kwargs)
        elif filename.endswith('.xyz'):
            self.write_to_xyz(filename, **kwargs)
        else:
            raise ValueError('Do not know how to write to file `%s`' % filename)
        
        
    def write_to_vtk(self, filename, header=None):
        """ write polyline to a vtk file """
        num_p = len(self.points)
        if header is None:
            header = "3d curve with %d points\n" % num_p
        with open(filename, 'w') as fp:
            fp.write("# vtk DataFile Version 2.0\n")
            fp.write("%s\n" % header)
            fp.write("ASCII\n\n")

            fp.write("DATASET POLYDATA\n")            
            fp.write("POINTS %d float\n" % num_p)
            np.savetxt(fp, self.points, delimiter=' ')
            fp.write("\n")
            
            fp.write("LINES 1 %d %d\n" % (num_p + 1, num_p))
            fp.write("\n".join(str(k) for k in xrange(num_p)))
            
            
    def write_to_xyz(self, filename, header=None, element='P'):
        """ write polyline to a xyz file """
        num_p = len(self.points)
        if header is None:
            header = "3d curve with %d points\n" % num_p
        with open(filename, 'w') as fp:
            fp.write("%d\n" % num_p)
            fp.write("%s\n" % header)
            for p in self.points:
                fp.write(element + ' {} {} {}\n'.format(*p))
                        
                        
                    
def _normalize_vectors(vectors):
    """ takes a list of vectors and normalizes them individually. """
    with np.errstate(divide='ignore', invalid='ignore'):
        return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

