'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

from .. import nested_dict



class TestNestedDict(unittest.TestCase):


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def test_simple(self):
        """ simple tests that reflect normal usage """
        d = nested_dict.NestedDict({'a': {'b': {}}})    
        self.assertEqual(d['a/b'], {})
        self.assertRaises(KeyError, lambda: d['c'])
        self.assertRaises(KeyError, lambda: d['a/c'])
        
        d['a/c'] = 2
        self.assertEqual(d['a/c'], 2)
        
        d['e/f'] = 3
        self.assertEqual(d['e/f'], 3)
        
        self.assertEqual(d.to_dict(), {'a': {'b': {}, 'c': 2}, 'e': {'f': 3}})
        self.assertEqual(d.to_dict(flatten=True), {'a/c': 2, 'e/f': 3})



if __name__ == "__main__":
    unittest.main()
    