'''
Created on Aug 29, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from .. import random


class TestRandom(unittest.TestCase):


    _multiprocess_can_split_ = True #< let nose know that tests can run parallel


    def test_random_log_uniform(self):
        """ tests the random_log_uniform function """
        data = random.log_uniform(v_min=1, v_max=10, size=1000)
        self.assertEqual(data.size, 1000)
        self.assertTrue(data.min() >= 1)
        self.assertTrue(data.max() <= 10)


    def test_take_combinations(self):
        """ tests the take_combinations function """
        num = 10
        data = np.arange(num)
        
        # test length of the result
        res = list(random.take_combinations(data, r=1, num=5))
        self.assertEqual(len(res), 5)
        res = list(random.take_combinations(data, r=1, num=2*num))
        self.assertEqual(len(res), num)
        res = list(random.take_combinations(data, r=1, num='all'))
        self.assertEqual(len(res), num)
        res = list(random.take_combinations(data, r=2, num='all'))
        self.assertEqual(len(res), num*(num - 1)//2)
        # test larger case where a different method is used
        res = list(random.take_combinations(data, r=3, num=10))
        self.assertEqual(len(res), 10)
        
        # test content of the result
        res = list(random.take_combinations(data, r=1, num=5))
        for value in res:
            self.assertIn(value, data)


    def test_take_product(self):
        """ test the take_product function """
        num = 5
        data = np.arange(num)
        res = list(random.take_product(data, r=1, num=5))
        self.assertEqual(len(res), 5)
        res = list(random.take_product(data, r=1, num=2*num))
        self.assertEqual(len(res), num)
        res = list(random.take_product(data, r=1, num='all'))
        self.assertEqual(len(res), num)
        res = list(random.take_product(data, r=2, num='all'))
        self.assertEqual(len(res), num**2)
        # test larger case where a different method is used
        res = list(random.take_product(data, r=3, num=10))
        self.assertEqual(len(res), 10)

        # test content of the result
        res = list(random.take_product(data, r=1, num=5))
        for value in res:
            self.assertIn(value, data)


        


if __name__ == "__main__":
    unittest.main()