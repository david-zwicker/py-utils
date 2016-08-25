'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

from ..misc import *  # @UnusedWildImport



class Test(unittest.TestCase):


    _multiprocess_can_split_ = True #< let nose know that tests can run parallel


    def test_transpose_list_of_dicts(self):
        """ test transpose_list_of_dicts function """
        data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 6}]
        res = transpose_list_of_dicts(data)
        self.assertEquals(res, {'a': [1, 5], 'b': [2, 6]})

        data = [{'a': 1, 'b': 2}, {'a': 5}]
        res = transpose_list_of_dicts(data)
        self.assertEquals(res, {'a': [1, 5], 'b': [2, None]})

        data = [{'a': 1, 'b': 2}, {'a': 5}]
        res = transpose_list_of_dicts(data, missing=-1)
        self.assertEquals(res, {'a': [1, 5], 'b': [2, -1]})



if __name__ == "__main__":
    unittest.main()