'''
Created on Nov 4, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy.spatial import distance
import networkx as nx

from . import to_array



def connect_components(graph, pos_attr, length_attr=None):
    """ connects all components by iteratively inserting edges between
    components that have minimal distance.
    
    `graph` is a networkx graph object with positions assigned to nodes
    `pos_attr` gives the key for the node attribute that stores the position
    `length_attr` stores the length of the new edges in this edge attribute
    """
    if nx.is_empty(graph):
        return graph
    
    # build a distance matrix for all nodes
    vertices = nx.get_node_attributes(graph, pos_attr)
    nodes = to_array(vertices.keys())
    positions = to_array(vertices.values())
    dists = distance.squareform(distance.pdist(positions))
        
    if len(vertices) != nx.number_of_nodes(graph):
        raise ValueError("Not all nodes have a position specified by the node "
                         "attribute `%s`" % pos_attr)

    # get all subgraphs and build a list of indices into the distance matrix
    try:
        # networkx version 2
        from networkx.algorithms.components import connected_components
    except ImportError:
        # networkx version 1
        from networkx import connected_component_subgraphs
    else:
        def connected_component_subgraphs(G):
            """ helper function generating the induced subgraphs """
            return (G.subgraph(c).copy() for c in connected_components(G))
    
    subgraphs = list(connected_component_subgraphs(graph))
    num_subgraphs = len(subgraphs)
    # find the index of each node of each subgraph in the nodes array
    sg_nids_list = [[np.flatnonzero(nodes == n)[0] for n in sg.nodes()]
                    for sg in subgraphs]
    
    assert sum(len(s) for s in sg_nids_list) == nx.number_of_nodes(graph)
    
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
            dist = dist_mat[x, y]
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
        if length_attr is not None:
            attr_dict = {length_attr: dists[nid_res, nid_sg]}
        else:
            attr_dict = None
        result.add_edge(nodes[nid_res], nodes[nid_sg], **attr_dict)
            
        # remove the subgraph from the to-do list
        result_nids.extend(sg_nids_list.pop(sg_min))
        del subgraphs[sg_min]
        
    assert nx.is_connected(result)
    assert nx.number_of_nodes(result) == len(vertices)
    assert nx.number_of_edges(result) == \
            nx.number_of_edges(graph) + num_subgraphs - 1
        
    return result

