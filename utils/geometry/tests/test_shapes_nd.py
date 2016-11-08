'''
Created on Nov 8, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from .. import shapes_nd


class TestShapesND(unittest.TestCase):

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel

    def test_project_point_on_line(self):
        l1 = [0, 0]
        l2 = [0, 1]
        
        p = shapes_nd.project_point_on_line([1, 1], l1, l2)
        np.testing.assert_almost_equal(p, [0, 1])
        p = shapes_nd.project_point_on_line([-1, -2], l1, l2)
        np.testing.assert_almost_equal(p, [0, -2])

        l1 = [0, 0]
        l2 = [1, 1]
        
        p = shapes_nd.project_point_on_line([1, 0], l1, l2)
        np.testing.assert_almost_equal(p, [0.5, 0.5])
        p = shapes_nd.project_point_on_line([-2, -2], l1, l2)
        np.testing.assert_almost_equal(p, [-2, -2])



if __name__ == "__main__":
    unittest.main()
