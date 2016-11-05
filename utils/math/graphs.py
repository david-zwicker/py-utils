'''
Created on Nov 4, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy.spatial import distance
import networkx as nx



def connect_components(graph, pos_key, dist_key=None):
    """ connects all components by iteratively inserting edges between
    components that have minimal distance.
    
    `graph` is a networkx graph object with positions assigned to nodes
    `pos_key` gives the key for the node attribute that stores the position
    `dist_key` if given, stores the length of the new edges as edge attribute
    """
    
    # build a distance matrix for all nodes
    vertices = nx.get_node_attributes(graph, pos_key)
    nodes = np.array(vertices.keys())
    positions = vertices.values()
    dists = distance.squareform(distance.pdist(positions))

    # get all subgraphs and build a list of indices into the distance matrix
    subgraphs = list(nx.connected_component_subgraphs(graph))
    sg_nids_list = []
    for sg in subgraphs:
        sg_nids = [np.flatnonzero(nodes == n)[0] for n in sg.nodes()]
        sg_nids_list.append(sg_nids)
    
    # initialize result with first subgraph
    result = subgraphs.pop(0)
    result_nids = sg_nids_list.pop(0)
    
    # iterate until all subgraphs have been added
    while subgraphs:
        # find subgraph that is closest to `result`
        sg_min, nid_sg, nid_res, dist_min = None, None, None, np.inf
        for k, sg_nids in enumerate(sg_nids_list):
            dist_mat = dists[sg_nids, :][:, result_nids]
            x, y = np.unravel_index(dist_mat.argmin(), dist_mat.shape)
            dist = dists[x, y]
            if dist < dist_min:
                sg_min = k  # index into the subgraph
                dist_min = dist  # its distance to `result`
                # store the node indices for the connecting edge
                nid_sg = sg_nids[x]
                nid_res = result_nids[y]

        # add graph to `result`
        result.add_nodes_from(subgraphs[sg_min].nodes(data=True))
        result.add_edges_from(subgraphs[sg_min].edges(data=True))
        
        # add a new edge between the subgraph and `result`
        if dist_key is not None:
            attr_dict = {dist_key: dists[nid_res, nid_sg]}
        else:
            attr_dict = None
        result.add_edge(nodes[nid_res], nodes[nid_sg], attr_dict)
            
        # remove the subgraph from the to-do list
        result_nids.extend(sg_nids_list.pop(sg_min))
        subgraphs.pop(sg_min)
        
    return result

