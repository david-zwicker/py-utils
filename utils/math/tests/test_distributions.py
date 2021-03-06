'''
Created on Aug 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import itertools
import unittest

import numpy as np
import scipy.stats

from .. import distributions


      
class TestMathDistributions(unittest.TestCase):
    """ unit tests for the continuous library """

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel
    
    
    def assertAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the arrays a and b """
        self.assertTrue(np.allclose(a, b, rtol, atol), msg)


    def test_dist_rvs(self):
        """ test random variates """
        # create some distributions to test
        distribution_list = [
            distributions.lognorm_mean(np.random.random() + 0.1,
                                       np.random.random() + 0.1),
            distributions.lognorm_mean_var(np.random.random() + 0.1,
                                           np.random.random() + 0.2),
            distributions.loguniform_mean(np.random.random() + 0.1,
                                          np.random.random() + 1.1),
        ]
        
        # calculate random variates and compare them to the given mean and var.
        for dist in distribution_list:
            rvs = dist.rvs(int(1e6))
            self.assertAllClose(dist.mean(), rvs.mean(), rtol=0.02,
                                msg='Mean of the distribution is not '
                                    'consistent.')
            self.assertAllClose(dist.var(), rvs.var(), rtol=0.4, atol=0.2,
                                msg='Variance of the distribution is not '
                                    'consistent.')
            
            
    def test_lognorm_mean_var_to_mu_sigma(self):
        """ test the lognorm_mean_var_to_mu_sigma function """
        m, v = 1, 1
        
        # test numpy definition
        mu, sigma = distributions.lognorm_mean_var_to_mu_sigma(m, v, 'numpy')
        xs = np.random.lognormal(mu, sigma, size=int(1e6))
        
        self.assertAlmostEqual(xs.mean(), m, places=2)
        self.assertAlmostEqual(xs.var(), v, places=1)
        
        # test scipy definition
        mu, sigma = distributions.lognorm_mean_var_to_mu_sigma(m, v, 'scipy')
        dist = scipy.stats.lognorm(scale=mu, s=sigma)
        
        self.assertAlmostEqual(dist.mean(), m, places=7)
        self.assertAlmostEqual(dist.var(), v, places=7)
        
        # additional parameter
        with self.assertRaises(ValueError):
            distributions.lognorm_mean_var_to_mu_sigma(m, v, 'non-sense')
            
            
    def test_lognorm_mean_var(self):
        """ test the lognorm_mean_var function """
        for mean, var in itertools.product((0.1, 1), (1, 0.1, 0)):
            dist = distributions.lognorm_mean_var(mean, var)
            self.assertAlmostEqual(dist.mean(), mean)
            self.assertAlmostEqual(dist.var(), var)
            
            
            
    def test_log_normal(self):
        """ test the log normal distribution """
        S0, sigma = np.random.random(2) + 0.1
        mu = S0 * np.exp(-0.5*sigma**2)
        var = S0**2 * (np.exp(sigma**2) - 1)
        
        # test our distribution and the scipy distribution
        dists = (distributions.lognorm_mean(S0, sigma),
                 scipy.stats.lognorm(scale=mu, s=sigma))
        for dist in dists:
            self.assertAlmostEqual(dist.mean(), S0)
            self.assertAlmostEqual(dist.var(), var)
        
        # test the numpy distribution
        rvs = np.random.lognormal(np.log(mu), sigma, size=int(1e6))
        self.assertAlmostEqual(rvs.mean(), S0, places=2)
        self.assertAlmostEqual(rvs.var(), var, places=1)

        # test the numpy distribution
        mean, var = np.random.random() + 0.1, np.random.random() + 0.1
        dist = distributions.lognorm_mean(mean, var)
        self.assertAlmostEqual(dist.mean(), mean)
        dist = distributions.lognorm_mean_var(mean, var)
        self.assertAlmostEqual(dist.mean(), mean)
        self.assertAlmostEqual(dist.var(), var)
        
        mu, sigma = distributions.lognorm_mean_var_to_mu_sigma(mean, var,
                                                                    'numpy')
        rvs = np.random.lognormal(mu, sigma, size=int(1e6))
        self.assertAlmostEqual(rvs.mean(), mean, places=2)
        self.assertAllClose(rvs.var(), var, rtol=0.4, atol=0.2)


    def test_gamma(self):
        """ test the log uniform distribution """
        mean = np.random.random() + 0.1
        var = np.random.random() + 1.1
        
        # test our distribution and the scipy distribution
        dist = distributions.gamma_mean_var(mean, var)
        self.assertAlmostEqual(dist.mean(), mean)
        self.assertAlmostEqual(dist.var(), var)
                    
                    
    def test_log_uniform(self):
        """ test the log uniform distribution """
        S0 = np.random.random() + 0.1
        width = np.random.random() + 1.1
        
        # test our distribution and the scipy distribution
        dist = distributions.loguniform_mean(S0, width)
        self.assertAlmostEqual(dist.mean(), S0)
        a, b = dist.support
        self.assertAlmostEqual(b / a, width**2)
        
        # test setting variance
        dist = distributions.loguniform_mean_var(S0, width)
        self.assertAlmostEqual(dist.mean(), S0)
        self.assertAlmostEqual(dist.var(), width)

        # test special case
        dist = distributions.loguniform_mean(S0, 1)
        self.assertAlmostEqual(dist.mean(), S0)
        self.assertEqual(dist.var(), 0)
        
        
    def test_deterministic_dist(self):
        val = np.random.random()
        dist = distributions.DeterministicDistribution(val)
        self.assertEqual(dist.mean(), val)
        self.assertEqual(dist.var(), 0)
        self.assertAllClose(dist.rvs(5), np.full(5, val))
                    
    

if __name__ == '__main__':
    unittest.main()
