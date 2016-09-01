'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np
import scipy.stats

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
            
            
    def test_lognorm_mean_var_to_mu_sigma(self):
        """ test the lognorm_mean_var_to_mu_sigma function """
        mean, var = 1, 1
        
        # test numpy definition
        mu, sigma = stats.lognorm_mean_var_to_mu_sigma(mean, var, 'numpy')
        xs = np.random.lognormal(mu, sigma, size=int(1e7))
        
        self.assertAlmostEqual(xs.mean(), mean, places=1)
        self.assertAlmostEqual(xs.var(), var, places=1)
        
        # test scipy definition
        mu, sigma = stats.lognorm_mean_var_to_mu_sigma(mean, var, 'scipy')
        dist = scipy.stats.lognorm(scale=mu, s=sigma)
        
        self.assertAlmostEqual(dist.mean(), mean, places=7)
        self.assertAlmostEqual(dist.var(), var, places=7)
        
        # additional parameter
        with self.assertRaises(ValueError):
            stats.lognorm_mean_var_to_mu_sigma(mean, var, 'non-sense')
            
            
    def test_lognorm_mean_var(self):
        """ test the lognorm_mean_var function """
        for mean, var in [(0.1, 1), (1, 0.1)]:
            dist = stats.lognorm_mean_var(mean, var)
            self.assertAlmostEqual(dist.mean(), mean)
            self.assertAlmostEqual(dist.var(), var)
            
            
    def _test_StatisticsAccumulator(self, shape=None, ddof=2):
        """ test the StatisticsAccumulator class """
        if shape is None:
            x = np.random.random(10)
        else:
            x = np.random.random([10] + shape)
        
        acc = stats.StatisticsAccumulator(shape=shape, ddof=ddof)
        acc.add_many(x)
        np.testing.assert_allclose(acc.mean, x.mean(axis=0))
        np.testing.assert_allclose(acc.std, x.std(axis=0, ddof=ddof))
    
    
    def test_StatisticsAccumulator(self):
        """ test the StatisticsAccumulator class """
        for ddof in [0, 2]:
            for shape in [None, [1], [3], [2, 3]]:
                self._test_StatisticsAccumulator(shape=shape, ddof=ddof)
        


if __name__ == "__main__":
    unittest.main()
