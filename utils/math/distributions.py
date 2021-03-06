'''
Created on Feb 24, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module provides functions and classes for probability distributions, which
build upon the scipy.stats package and extend it.   
'''

from __future__ import division

import numpy as np
from scipy import stats, special, linalg, optimize

from ..data_structures.cache import cached_property



def lognorm_mean_var_to_mu_sigma(mean, variance, definition='scipy'):
    """ determines the parameters of the log-normal distribution such that the
    distribution yields a given mean and variance. The optional parameter
    `definition` can be used to choose a definition of the resulting parameters
    that is suitable for the given software package. """
    mean2 = mean**2
    mu = mean2/np.sqrt(mean2 + variance)
    sigma = np.sqrt(np.log(1 + variance/mean2))
    if definition == 'scipy':
        return mu, sigma
    elif definition == 'numpy':
        return np.log(mu), sigma
    else:
        raise ValueError('Unknown definition `%s`' % definition)



def lognorm_mean(mean, sigma):
    """ returns a lognormal distribution parameterized by its mean and a spread
    parameter `sigma` """
    if sigma == 0:
        return DeterministicDistribution(mean)
    else:
        mu = mean * np.exp(-0.5 * sigma**2)
        return stats.lognorm(scale=mu, s=sigma)



def lognorm_mean_var(mean, variance):
    """ returns a lognormal distribution parameterized by its mean and its
    variance. """
    if variance == 0:
        return DeterministicDistribution(mean)
    else:
        scale, sigma = lognorm_mean_var_to_mu_sigma(mean, variance, 'scipy')
        return stats.lognorm(scale=scale, s=sigma)



def lognorm_sum_leastsq(count, var_norm, sim_terms=1e5, bins=64):
    """ returns the parameters of a log-normal distribution that estimates the
    sum of `count` log-normally distributed random variables with mean 1 and
    variance `var_norm`. These parameters are determined by fitting the 
    probability density function to a histogram obtained by drawing `sim_terms`
    random numbers """
    sum_mean = count
    sum_var = count * var_norm
    
    # get random numbers
    dist = lognorm_mean_var(1, var_norm)
    vals = dist.rvs((int(sim_terms), count)).sum(axis=1)
    
    # get the histogram
    val_max = sum_mean + 3 * np.sqrt(sum_var)
    bins = np.linspace(0, val_max, bins + 1)
    xs = 0.5*(bins[:-1] + bins[1:])
    density, _ = np.histogram(vals, bins=bins, range=[0, val_max],
                              density=True)
    
    def pdf_diff(params):
        """ evaluate the estimated pdf """
        scale, sigma = params
        return stats.lognorm.pdf(xs, scale=scale, s=sigma) - density
        
    # do the least square fitting
    params_init = lognorm_mean_var_to_mu_sigma(sum_mean, sum_var, 'scipy')
    params, _ = optimize.leastsq(pdf_diff, params_init)
    return params



def lognorm_sum(count, mean, variance, method='fenton'):
    """ returns an estimate of the distribution of the sum of `count`
    log-normally distributed variables with `mean` and `variance`. The returned
    distribution is again log-normal with mean and variance determined from the
    given parameters. Here, several methods can be used:
        `fenton` - match the first two moments of the distribution
        `leastsq` - minimize the error in the interval 
    
    """
    if method == 'fenton':
        # use the moments directly
        return lognorm_mean_var(count * mean, count * variance)
        
    elif method == 'leastsq':
        # determine the moments from fitting
        var_norm = variance / mean**2
        scale, sigma = lognorm_sum_leastsq(count, var_norm)
        return stats.lognorm(scale=scale * mean, s=sigma)
        
    else:
        raise ValueError('Unknown method `%s` for determining the sum of '
                         'lognormal distributions. Accepted methods are '
                         '[`fenton`, `leastsq`].')
    


def gamma_mean_var(mean, variance):
    """ returns a gamma distribution with given mean and variance """
    alpha = mean**2 / variance
    beta = variance / mean
    return stats.gamma(scale=beta, a=alpha)



def loguniform_mean(mean, width):
    """ returns a loguniform distribution parameterized by its mean and a spread
    parameter `width`. The ratio between the maximal value and the minimal value
    is given by width**2 """
    if width == 1:
        # treat special case separately
        return DeterministicDistribution(mean)
    else:
        scale = mean * (2*width*np.log(width)) / (width**2 - 1)
        return LogUniformDistribution(scale=scale, s=width)



def loguniform_mean_var(mean, var):
    """ returns a loguniform distribution parameterized by its mean and
    variance. Here, we need to solve a non-linear equation numerically, which
    might degrade accuracy and performance of the result """
    if var < 0:
        raise ValueError('Variance must be positive')
    elif var == 0:
        # treat special case separately
        return DeterministicDistribution(mean)
    else:
        # determine width parameter numerically
        cv2 = var / mean**2  # match square coefficient of variation
        
        def _rhs(q):
            """ match the coefficient of variation """
            return 0.5 * (q + 1) * np.log(q) / (q - 1) - 1 - cv2
        
        width = optimize.newton(_rhs, 1.1)
        return loguniform_mean(mean, np.sqrt(width))



def random_log_uniform(v_min, v_max, size):
    """ returns random variables that a distributed uniformly in log space """
    log_min, log_max = np.log(v_min), np.log(v_max)
    res = np.random.uniform(log_min, log_max, size)
    return np.exp(res)



def dist_skewness(dist):
    """ returns the skewness of the distribution `dist` """
    mean = dist.mean()
    var = dist.var()
    return (dist.moment(3) - 3*mean*var - mean**3) / var**(3/2)



class DeterministicDistribution_gen(stats.rv_continuous):
    """ deterministic distribution that always returns a given value
    Code copied from
    https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html#making-a-continuous-distribution-i-e-subclassing-rv-continuous
    """
    def _cdf(self, x):
        return np.where(x < 0, 0., 1.)
    
    def _stats(self):
        return 0., 0., 0., 0.
    
    def _rvs(self):
        return np.zeros(self._size)
    


DeterministicDistribution = DeterministicDistribution_gen(
    name='DeterministicDistribution'
)



class LogUniformDistribution_gen(stats.rv_continuous):
    """
    Log-uniform distribution.
    """ 
    
    def freeze(self, *args, **kwds):
        frozen = super(LogUniformDistribution_gen, self).freeze(*args, **kwds)
        frozen.support = self.support(*args, **kwds)
        return frozen

    
    def support(self, *args, **kwds):
        """ return the interval in which the PDF of the distribution is
        non-zero """
        extra_args, _, _, _ = self._parse_args_stats(*args, **kwds)
        mean = self.mean(*args, **kwds)
        scale = extra_args[0]
        width = mean * (2*scale*np.log(scale)) / (scale**2 - 1)
        return (width / scale, width * scale)
    

    def _rvs(self, s):
        """ random variates """
        # choose the receptor response characteristics
        return random_log_uniform(1/s, s, self._size)
    
    
    def _pdf(self, x, s):
        """ probability density function """
        s = s[0]  # reset broadcasting
        res = np.zeros_like(x)
        idx = (1 < x*s) & (x < s)
        res[idx] = 1/(x[idx] * np.log(s*s))
        return res
        
        
    def _cdf(self, x, s): 
        """ cumulative probability function """
        s = s[0]  # reset broadcasting
        res = np.zeros_like(x)
        idx = (1 < x*s) & (x < s)
        log_s = np.log(s)
        res[idx] = (log_s + np.log(x[idx]))/(2 * log_s)
        res[x > s] = 1
        
        return res


    def _ppf(self, q, s):
        """ percent point function (inverse of cdf) """
        s = s[0]  # reset broadcasting
        res = np.zeros_like(q)
        idx = (q > 0)
        res[idx] = s**(2*q[idx] - 1)
        return res
    
    
    def _stats(self, s):
        """ calculates statistics of the distribution """
        mean = (s**2 - 1)/(2*s*np.log(s))
        var = ((s**4 - 1) * np.log(s) - (s**2 - 1)**2) \
                / (4 * s**2 * np.log(s)**2)
        return mean, var, None, None

    

LogUniformDistribution = LogUniformDistribution_gen(
    a=0, name='LogUniformDistribution'
)



class HypoExponentialDistribution(object):
    """
    Hypoexponential distribution.
    Unfortunately, the framework supplied by scipy.stats.rv_continuous does not
    support a variable number of parameters and we thus only mimic its
    interface here.
    """
    
    def __init__(self, rates, method='sum'):
        """ initializes the hypoexponential distribution.
        `rates` are the rates of the underlying exponential processes
        `method` determines what method is used for calculating the cdf and can
            be either `sum` or `eigen`        
        """
        if method in {'sum', 'eigen'}:
            self.method = method
        
        # prepare the rates of the system
        self.rates = np.asarray(rates)
        self.alpha = 1 / self.rates
        if np.any(rates <= 0):
            raise ValueError('All rates must be positive')
        if len(np.unique(self.alpha)) != len(self.alpha):
            raise ValueError('The current implementation only supports cases '
                             'where all rates are different from each other.')
        
        # calculate terms that we need later
        with np.errstate(divide='ignore'):
            mat = self.alpha[:, None] \
                    / (self.alpha[:, None] - self.alpha[None, :])
        mat[(self.alpha[:, None] - self.alpha[None, :]) == 0] = 1
        self._terms = np.prod(mat, 1)
        
    
    def rvs(self, size):
        """ random variates """
        # choose the receptor response characteristics
        return sum(np.random.exponential(scale=alpha, size=size)
                   for alpha in self.alpha)

    
    def mean(self):
        """ mean of the distribution """
        return self.alpha.sum()
    
    
    def variance(self):
        """ variance of the distribution """
        return (2 * np.sum(self.alpha**2 * self._terms) - 
                (self.alpha.sum())**2)
    

    def pdf(self, x):
        """ probability density function """
        if not np.isscalar(x):
            x = np.asarray(x)
            res = np.zeros_like(x)
            nz = (x > 0)
            if np.any(nz):
                if self.method == 'sum':
                    factor = np.exp(-x[nz, None] * self.rates[..., :]) \
                                / self.rates[..., :]
                    res[nz] = np.sum(self._terms[..., :] * factor, axis=1)
                else:
                    Theta = (np.diag(-self.rates, 0) + 
                             np.diag(self.rates[:-1], 1))
                    for i in np.flatnonzero(nz):
                        res.flat[i] = \
                                1 - linalg.expm(x.flat[i]*Theta)[0, :].sum()
 
        elif x == 0:
            res = 0
        else:
            if self.method == 'sum':
                factor = np.exp(-x*self.rates)/self.ratesx
                res[nz] = np.sum(self._terms * factor)
            else:
                Theta = np.diag(-self.rates, 0) + np.diag(self.rates[:-1], 1)
                res = 1 - linalg.expm(x*Theta)[0, :].sum()
        return res

    
    def cdf(self, x):
        """ cumulative density function """
        if not np.isscalar(x):
            x = np.asarray(x)
            res = np.zeros_like(x)
            nz = (x > 0)
            if np.any(nz):
                factor = np.exp(-x[nz, None]*self.rates[..., :])
                res = 1 - np.sum(self._terms[..., :] * factor, axis=1)
        elif x == 0:
            res = 0
        else:
            factor = np.exp(-x*self.rates)
            res = 1 - np.sum(self._terms * factor)
        return res           



# ==============================================================================
# OLD DISTRIBUTIONS THAT MIGHT NOT BE NEEDED ANYMORE
# ==============================================================================



class PartialLogNormDistribution_gen(stats.rv_continuous):
    """
    partial log-normal distribution.
    a fraction `frac` of the distribution follows a log-normal distribution,
    while the remaining fraction `1 - frac` is zero
    
    Similar to the lognorm distribution, this does not support any location
    parameter
    """ 
    
    def _rvs(self, s, frac):
        """ random variates """
        # choose the items response characteristics
        res = np.exp(s * np.random.standard_normal(self._size))
        if frac != 1:
            # switch off items randomly
            res[np.random.random(self._size) > frac] = 0
        return res
    
    
    def _pdf(self, x, s, frac):
        """ probability density function """
        s, frac = s[0], frac[0]  # reset broadcasting
        return frac / (s*x*np.sqrt(2*np.pi)) * np.exp(-1/2*(np.log(x)/s)**2)         
        
        
    def _cdf(self, x, s, frac): 
        """ cumulative probability function """
        s, frac = s[0], frac[0]  # reset broadcasting
        return 1 + frac*(-0.5 + 0.5*special.erf(np.log(x)/(s*np.sqrt(2))))


    def _ppf(self, q, s, frac):
        """ percent point function (inverse of cdf) """
        s, frac = s[0], frac[0]  # reset broadcasting
        q_scale = (q - (1 - frac)) / frac
        res = np.zeros_like(q)
        idx = (q_scale > 0)
        res[idx] = np.exp(s * special.ndtri(q_scale[idx]))
        return res


PartialLogNormDistribution = PartialLogNormDistribution_gen(
    a=0, name='PartialLogNormDistribution'
)




class PartialLogUniformDistribution_gen(stats.rv_continuous):
    """
    partial log-uniform distribution.
    a fraction `frac` of the distribution follows a log-uniform distribution,
    while the remaining fraction `1 - frac` is zero
    """ 
    
    def _rvs(self, s, frac):
        """ random variates """
        # choose the receptor response characteristics
        res = random_log_uniform(1/s, s, self._size)
        # switch off receptors randomly
        if frac != 1:
            res[np.random.random(self._size) > frac] = 0
        return res
    
    
    def _pdf(self, x, s, frac):
        """ probability density function """
        s, frac = s[0], frac[0]  # reset broadcasting
        res = np.zeros_like(x)
        idx = (1 < x*s) & (x < s)
        res[idx] = frac/(x[idx] * np.log(s*s))
        return res         
        
        
    def _cdf(self, x, s, frac): 
        """ cumulative probability function """
        s, frac = s[0], frac[0]  # reset broadcasting
        res = np.zeros_like(x)
        idx = (1 < x*s) & (x < s)
        log_s = np.log(s)
        res[idx] = (log_s + np.log(x[idx]))/(2 * log_s)
        res[x > s] = 1
        
        return (1 - frac) + frac*res


    def _ppf(self, q, s, frac):
        """ percent point function (inverse of cdf) """
        s, frac = s[0], frac[0]  # reset broadcasting
        q_scale = (q - (1 - frac)) / frac
        res = np.zeros_like(q)
        idx = (q_scale > 0)
        res[idx] = s**(2*q_scale[idx] - 1)
        return res
    

PartialLogUniformDistribution = PartialLogUniformDistribution_gen(
    a=0, name='PartialLogUniformDistribution'
)



NORMAL_DISTRIBUTION_NORMALIZATION = 1/np.sqrt(2*np.pi)


class NormalDistribution(object):
    """ class representing normal distributions """ 
    
    def __init__(self, mean, var, count=None):
        """ normal distributions are described by their mean and variance.
        Additionally, count denotes how many observations were used to
        estimate the parameters. All values can also be numpy arrays to
        represent many distributions efficiently """ 
        self.mean = mean
        self.var = var
        self.count = count
        
        
    def copy(self):
        return self.__class__(self.mean, self.var, self.count)
        
        
    @cached_property()
    def std(self):
        """ return standard deviation """
        return np.sqrt(self.var)
        
    
    def pdf(self, value, mask=None):
        """ return probability density function at value """
        if mask is None:
            mean = self.mean
            var = self.var
            std = self.std
        else:
            mean = self.mean[mask]
            var = self.var[mask]
            std = self.std[mask]
        
        return NORMAL_DISTRIBUTION_NORMALIZATION/std \
                * np.exp(-0.5*(value - mean)**2 / var)
                
                
    def add_observation(self, value):
        """ add an observed value and adjust mean and variance of the
        distribution. This returns a new distribution and only works if
        count was set """
        if self.count is None:
            return self.copy()
        else:
            M2 = self.var*(self.count - 1)
            count = self.count + 1
            delta = value - self.mean
            mean = self.mean + delta/count
            M2 = M2 + delta*(value - mean)
            return NormalDistribution(mean, M2/(count - 1), count)
                
                
    def distance(self, other, kind='kullback-leibler'):
        """ return the distance between two normal distributions """
        if kind == 'kullback-leibler':
            dist = 0.5*(np.log(other.var/self.var) + 
                        (self.var + (self.mean - self.mean)**2)/other.var - 1) 
            
        elif kind == 'bhattacharyya':
            var_ratio = self.var/other.var
            term1 = np.log(0.25*(var_ratio + 1/var_ratio + 2))
            term2 = (self.mean - other.mean)**2/(self.var + other.var)
            dist = 0.25*(term1 + term2)
            
        elif kind == 'hellinger':
            dist_b = self.distance(other, kind='bhattacharyya')
            dist = np.sqrt(1 - np.exp(-dist_b))
            
        else:
            raise ValueError('Unknown distance `%s`' % kind)
        
        return dist
    
    
    def welch_test(self, other):
        """ performs Welch's t-test of two normal distributions """
        # calculate the degrees of freedom
        s1, s2 = self.var/self.count, other.var/other.count
        nu1, nu2 = self.count - 1, other.count - 1
        dof = (s1 + s2)**2/(s1**2/nu1 + s2**2/nu2)

        # calculate the Welch t-value
        t = (self.mean - other.mean)/np.sqrt(s1 + s2)
        
        # calculate the probability using the Student's T distribution 
        prob = stats.t.sf(np.abs(t), dof) * 2
        return prob
    
    
    def overlap(self, other, common_variance=True):
        """ estimates the amount of overlap between two distributions """
        if common_variance:
            if self.count is None:
                if other.count is None:  # neither is sampled
                    S = np.sqrt(0.5*(self.var + other.var))
                else:  # other is sampled
                    S = self.std
            else: 
                if other.count is None:  # self is sampled
                    S = other.std
                else:  # both are sampled
                    expr = ((self.count - 1)*self.var +
                            (other.count - 1)*other.var)
                    S = np.sqrt(expr/(self.count + other.count - 2))

            delta = np.abs(self.mean - other.mean)/S
            return 2*stats.norm.cdf(-0.5*delta)

        else:
            # here, we would have to integrate numerically
            raise NotImplementedError
