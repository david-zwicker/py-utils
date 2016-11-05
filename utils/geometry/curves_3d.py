'''
Created on Nov 4, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

import itertools

import numpy as np
from scipy.spatial import distance

from utils.data_structures.cache import cached_property



class Curve3D(object):
    ''' represents a curve in 3d space '''


    def __init__(self, points):
        ''' the curve is given by a collection of linear segments '''
        self.points = np.atleast_2d(points)
        if self.points.size > 0 and self.points.shape[1] != 3:
            raise ValueError('points must be a nx3 array.')
        
    
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
            