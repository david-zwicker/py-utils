'''
Created on Nov 4, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

import itertools

import numpy as np
from scipy import interpolate
from scipy.spatial import distance

from utils.data_structures.cache import cached_property



class Curve3D(object):
    ''' represents a curve in 3d space '''


    def __init__(self, points):
        ''' the curve is given by a collection of linear segments '''
        self.points = points
        if self.points.size > 0 and self.points.shape[1] != 3:
            raise ValueError('points must be a nx3 array.')


    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, points):
        self._points = np.atleast_2d(points)
        self._cache_properties = {}
        
    
    @cached_property
    def length(self):
        """ returns the length of the curve """
        return sum(distance.euclidean(p1, p2)
                   for p1, p2 in itertools.izip(self.points, self.points[1:]))
    
        
    def iter(self, with_normals=False):
        """ iterates over the points and returns their coordinates
        `with_normals` also returns the local normal vector         
        """
        if with_normals:
            # return the points and the normals
            normals = np.gradient(self.points, axis=0)
            for p, n in itertools.izip(self.points, normals):
                yield p, n
                    
        else:
            # only return the points, not the normals
            for p in self.points:
                yield p 
                
        
    def __iter__(self):
        return self.iter()
    
    
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
            for p1, p2 in itertools.izip(self.points[:-1], self.points[1:]):
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
                
            # get arc length of support points
            segments = itertools.izip(self.points, self.points[1:])
            s = np.cumsum([np.linalg.norm(p2 - p1)
                           for p1, p2 in segments])
            s = np.insert(s, 0, 0) # prepend element for first point
            # divide arc length equidistantly
            sp = np.linspace(s[0], s[-1], count)
            # interpolate points
            result = np.transpose((np.interp(sp, s, self.points[:, 0]),
                                   np.interp(sp, s, self.points[:, 1]),
                                   np.interp(sp, s, self.points[:, 2])))
            
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
            self.points = zip(*points)  # transpose list
            
    
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
            