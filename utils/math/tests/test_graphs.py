'''
Created on Nov 4, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import networkx as nx
import six

from .. import graphs



class TestGraphs(unittest.TestCase):

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel

    def test_conncet_components(self):
        g = nx.Graph()
        
        g.add_node(0, pos=(0, 0))
        g.add_node(1, pos=(1, 1))
        g.add_edges_from([[0, 1]])
        
        # trivial test with connected graph
        gc = graphs.connect_components(g, 'pos', 'dist')
        self.assertTrue(nx.is_connected(gc))
        six.assertCountEqual(self, gc.nodes(), [0, 1])
        six.assertCountEqual(self, gc.edges(), [(0, 1)])
        
        # add another component
        g.add_node(10, pos=(10, 0))
        g.add_node(11, pos=(9, 1))
        g.add_edges_from([[10, 11]])
        
        gc = graphs.connect_components(g, 'pos', 'dist')
        self.assertTrue(nx.is_connected(gc))
        six.assertCountEqual(self, gc.nodes(), [0, 1, 10, 11])
        six.assertCountEqual(self, gc.edges(), [(0, 1), (10, 11), (1, 11)])
        self.assertEqual(nx.get_edge_attributes(gc, 'dist')[(1, 11)], 8)


    def test_conncet_components_error(self):
        g = nx.Graph()
        gc = graphs.connect_components(g, 'pos')
        self.assertTrue(nx.is_empty(gc))
        
        g.add_node(0)
        g.add_node(1, pos=(1, 1))
        g.add_edge(0, 1)
        self.assertRaises(ValueError,
                          lambda: graphs.connect_components(g, 'pos'))
        


if __name__ == "__main__":
    unittest.main()
