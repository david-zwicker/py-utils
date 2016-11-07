'''
Created on Nov 4, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

import numpy as np
from scipy import interpolate
from scipy.ndimage import filters

from six.moves import zip

from utils.data_structures.cache import cached_property



class Curve3D(object):
    ''' represents a curve in 3d space '''


    def __init__(self, points, curvature_smoothing=0):
        ''' the curve is given by a collection of linear segments
        
        `points` are the support points defining the curve
        `curvature_smoothing` gives the length scale over which calculated
            quantities, like tangent vectors, are smoothed. The default value
            of zero means no smoothing.
        '''
        self.points = points
        self.curvature_smoothing = curvature_smoothing
        if self.points.size > 0 and self.points.shape[1] != 3:
            raise ValueError('points must be a nx3 array.')


    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, points):
        self._points = np.atleast_2d(points)
        if self._points.size > 0 and self._points.shape[-1] != 3:
            raise ValueError('Coordinates must be 3-dimensional.')
        # clear cache
        self._cache_properties = {}
    
    
    @property
    def curvature_smoothing(self):
        return self._curvature_smoothing
    
    @curvature_smoothing.setter
    def curvature_smoothing(self, curvature_smoothing):
        self._curvature_smoothing = curvature_smoothing
        # clear cache
        self._cache_properties = {}
        
    
    @cached_property
    def length(self):
        """ returns the length of the curve """
        return np.linalg.norm(self.points[:-1] - self.points[1:], axis=1).sum()
    
    
    def smooth_normalized(self, vectors):
        """ takes a list of vectors and normalizes them individually. If one of
        the vectors is zero (and thus cannot be normalized) it is calculated
        from the average of the neighboring vectors """
        smoothing = self.curvature_smoothing
        if smoothing > 0:
            vectors = filters.gaussian_filter1d(vectors, sigma=smoothing,
                                                axis=0)
            
        with np.errstate(divide='ignore', invalid='ignore'):
            return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

    
    @cached_property
    def tangents(self):
        """ return the tangent vector at each support point """
        tangents = np.gradient(self.points, axis=0)
        return self.smooth_normalized(tangents)
    
    
    @cached_property
    def normals(self):
        """ return the normal vector at each support point """
        normals = np.gradient(self.tangents, axis=0)
        return self.smooth_normalized(normals)


    @cached_property
    def binormals(self):
        """ return the binormal vector at each support point """
        binormals = np.cross(self.tangents, self.normals)
        return self.smooth_normalized(binormals)
    
    
    @cached_property
    def stretching_factors(self):
        """ return the stretching factor at each support point. A stretching
        factor of 1 indicates an arc-length parametrization of the curve """
        tangent = np.gradient(self.points, axis=0)
        return np.linalg.norm(tangent, axis=-1)
    
    
    @cached_property
    def arc_lengths(self):
        """ return the arc length up to each support point """
        ds = np.linalg.norm(self.points[:-1] - self.points[1:], axis=1)
        return np.r_[0, np.cumsum(ds)]
    
    
    @cached_property
    def curvatures(self):
        """ return the curvature at each support point """
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
            curv = np.arccos(cos_a) * 2 / (n1 + n2)
            
        # the curvature of the end points are zero by definition 
        return np.r_[0, curv, 0]
        
        
    def iter(self, data=None, smoothing=0):
        """ iterates over the points and returns their coordinates
        
        `data` lists extra quantities that are returned for each point. Possible
            values include ('tangent', 'normal', 'binormal', 'unit_vectors',
            'curvature', 'arc_length', 'local_arc_length')
        `smoothing` defines a smoothing applied to the (discrete) derivatives
            
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
                        'curvature', 'arc_length', 'stretching_factor'}
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

            # TODO: Implement Torsion

            # return the requested data
            for n, p in enumerate(self.points):
                yield p, {k: calculated[k][n] for k in data}
                
        
    def __iter__(self):
        return self.iter()
    
    
    def invert_parameterization(self):
        """ inverts the parameterization """
        self.points = self.points[::-1]
    
    
    def make_equidistant(self, spacing=None, count=None):
        """ returns a new parameterization of the same curve where points have
        been chosen equidistantly. The original curve may be slightly modified
        
        `spacing` gives the approximate spacing between support points
        `count` gives the approximate number of support points
        
        Only either one of these parameters may be supplied
        """
        if spacing is not None:
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
            # interpolate points
            result = np.c_[np.interp(sp, s, ps[:, 0]),
                           np.interp(sp, s, ps[:, 1]),
                           np.interp(sp, s, ps[:, 2])]
            
        self.points = result
        
    
    def smooth(self, smoothing=10, degree=3, derivative=0, num_points=None):
        """ smooths the curve by interpolating the points
        
        `smoothing` determines the smoothness of the curve.  This value can be
            used to control the trade-off between closeness and smoothness of
            fit. Larger values means more smoothing while smaller values
            indicate less smoothing. The resulting, smoothed yi fulfill
                sum((y - yi)**2, axis=0) <= smoothing*len(points)
        `degree` determines the degree of the splines used
        `derivative` determines the order of the derivative
        `num_points` determines how many support points are used. If this value
            is None, len(points) are used.
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
            if num_points != len(self.points):
                self.make_equidistant(count=num_points)
        else:
            # interpolate the line
            points = interpolate.splev(u, tck, der=derivative)
            self.points = np.transpose(points)
            
    
    def write_to_file(self, filename):
        """ write the polyline to a file. The format is chosen automatically
        based on the file extension """
        if filename.endswith('.vtk'):
            self.write_to_vtk(filename)
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
            