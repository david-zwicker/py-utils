'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np
import six

from .. import stats



class TestStats(unittest.TestCase):
    """ test suite for statistics functions """

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel

    def test_mean_std_online(self):
        """ test the mean_std_online function """
        x = np.random.random(10)
        
        mean, std = stats.mean_std_online(x)
        self.assertAlmostEqual(mean, x.mean())
        self.assertAlmostEqual(std, x.std())
        
        mean, std = stats.mean_std_online(iter(x))
        self.assertAlmostEqual(mean, x.mean())
        self.assertAlmostEqual(std, x.std())

        mean, std = stats.mean_std_online(x, ddof=2)
        self.assertAlmostEqual(mean, x.mean())
        self.assertAlmostEqual(std, x.std(ddof=2))
        
        # corner cases
        mean, std = stats.mean_std_online([1])
        self.assertEqual(mean, 1)
        self.assertEqual(std, 0)
        
        mean, std = stats.mean_std_online([1], ddof=2)
        self.assertEqual(mean, 1)
        self.assertTrue(np.isnan(std))
        
        mean, std = stats.mean_std_online([])
        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(std))
        
        
    def test_mean_std_frequency_table(self):
        """ test the mean_std_frequency_table function """
        x = np.random.randint(0, 5, 10)
        f = np.bincount(x)
        for ddof in (0, 2):
            mean, std = stats.mean_std_frequency_table(f, ddof=ddof)
            self.assertAlmostEqual(mean, x.mean())
            self.assertAlmostEqual(std, x.std(ddof=ddof))

            
    def _test_StatisticsAccumulator(self, shape=None, ddof=2):
        """ test the StatisticsAccumulator class """
        if shape is None:
            x = np.random.random(10)
        else:
            x = np.random.random([10] + shape)
        
        acc = stats.StatisticsAccumulator(shape=shape, ddof=ddof)
        self.assertIsInstance(str(acc), six.string_types)
        
        acc.add(x[0])
        self.assertIsInstance(str(acc), six.string_types)

        acc.add_many(x[1:])
        np.testing.assert_allclose(acc.mean, x.mean(axis=0))
        np.testing.assert_allclose(acc.std, x.std(axis=0, ddof=ddof))
        
        self.assertIsInstance(str(acc), six.string_types)
        
        try:
            import uncertainties
        except ImportError:
            pass
        else:
            if shape is None:
                self.assertIsInstance(acc.to_uncertainties(),
                                      uncertainties.core.Variable)
            else:
                self.assertIsInstance(acc.to_uncertainties(), np.ndarray)
    
    
    def test_StatisticsAccumulator(self):
        """ test the StatisticsAccumulator class """
        for ddof in [0, 2]:
            for shape in [None, [1], [3], [2, 3]]:
                self._test_StatisticsAccumulator(shape=shape, ddof=ddof)
        


if __name__ == "__main__":
    unittest.main()
