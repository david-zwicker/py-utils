'''
Created on Dec 1, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import itertools

import numpy as np
from scipy import spatial

from six.moves import range



def _iter_problematic_edges(edges):
    """ generator that goes through a list of edges and returns intervals that
    are problematic, i.e., where the edge would cross a previous edge. Yields
    pairs of indices that denote the indices of the edges that surround a
    problem region. """ 
    i = 1
    while i < len(edges):
        if edges[i] < edges[i - 1]:
            # found problem
            s = i - 1
            e_s = edges[s]
            i += 1
            for j in range(i, len(edges)):
                if edges[j] >= e_s:
                    # problem ends
                    yield s, j
                    i = j
                    break
            else:
                # problem did not end
                if any(e > 0 for e in edges[s + 1:]):
                    yield s, len(edges)
        else:
            i += 1
            
            
            
def _fix_edge_order(edges, p_f, p_t, dists_ft):
    """ reassigns edges such that they don't cross and have minimal length """
    dim_f, dim_t = len(p_f), len(p_t)
    assert dists_ft.shape == (dim_f, dim_t)
    assert len(edges) == len(p_f)
    
    # iterate over all problematic regions and try to resolve them
    for f_s, f_e in _iter_problematic_edges(edges):
        # f_s is the index of the last correct vertex
        # f_e is the index of the first correct vertex after the problem
        
        # get candidate connections
        if f_e < dim_f:
            ts = np.arange(edges[f_s], edges[f_e] + 1)
        else:
            ts = np.r_[np.arange(edges[f_s], dim_t), 0]
        subdists = dists_ft[f_s+1:f_e, ts]
        num_f = f_e - f_s - 1  # number of vertices to correct
        num_t = len(ts)  # number of vertices to choose from
        assert subdists.shape == (num_f, num_t)
        
        # iterate through all possibilities
        best_offset, best_dist = None, np.inf
        for offset in itertools.product(range(num_t), repeat=num_f):
            d_offset = np.diff(offset)
            if np.any(d_offset < 0):
                continue
            dist = sum(subdists[i, offset[i]] for i in range(num_f))
            if dist < best_dist:
                best_dist = dist
                best_offset = offset
                
        # shift by 1 because np.diff shifted indices
        new_t = ts[0] + best_offset
        new_t[new_t >= dim_t] = 0
        edges[f_s + 1:f_e] = new_t
        
    return edges
        


def register_polygons(coords1, coords2):
    """ returns oriented edges between two polygons that are given by their
    coordinate sequences. 
    
    The function returns two list describing the edges that connect the two
    polygons. The first list is of length `len(coords1)` and gives for each
    vertex of the first polygon an index into `coords2` to describe the edge.
    The second list is the associated list for edges from polygon 2 to 1.  
    
    
    The algorithm tries to produce non-crossing edges with minimal length, which
    could be useful for morphing one polygon into the other. This algorithm
    works best when the centroids are close to each other.    
    """
    p1 = np.array(coords1, dtype=np.double, copy=True)
    p2 = np.array(coords2, dtype=np.double, copy=True)
    
    # build the distance matrix between the two polygons
    dists = spatial.distance_matrix(coords1, coords2)

    # find the points with minimal distance from each other 
    x, y = np.nonzero(dists == np.min(dists))
    x, y = x[0], y[0]
    
    # reorder the points such that the closest points are in the first position
    p1 = np.roll(p1, -x, axis=0)
    p2 = np.roll(p2, -y, axis=0)
    dists = np.roll(np.roll(dists, -x, axis=0), -y, axis=1)
    
    # assert np.all(dists == spatial.distance_matrix(p1, p2))
    
    # potential edges from 1 to 2 and vice versa
    e12 = np.argmin(dists, axis=1)
    e21 = np.argmin(dists, axis=0)
    
    # fix edge order in place
    _fix_edge_order(e12, p1, p2, dists)
    _fix_edge_order(e21, p2, p1, dists.T)
    
    # roll back the edges so they align with the original coordinates
    e12 = np.roll(e12, x)
    e12 = (e12 + y) % len(p2)
    
    e21 = np.roll(e21, y)
    e21 = (e21 + x) % len(p1)
    
    return e12, e21



