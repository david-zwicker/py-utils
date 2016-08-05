'''
Created on Feb 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains math functions
'''

from __future__ import division


import numpy as np
from scipy import interpolate



def xlog2x(x):
    """ calculates x*np.log2(x) """
    if x == 0:
        return 0
    else:
        return x * np.log2(x)

# vectorize the function above
xlog2x = np.vectorize(xlog2x, otypes='d')



def arrays_close(arr1, arr2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """ compares two arrays using a relative and an absolute tolerance """
    arr1 = np.atleast_1d(arr1)
    arr2 = np.atleast_1d(arr2)
    
    if arr1.shape != arr2.shape:
        # arrays with different shape are always unequal
        return False
        
    if equal_nan:
        # skip entries where both arrays are nan
        idx = ~(np.isnan(arr1) & np.isnan(arr2))
        if idx.sum() == 0:
            # occurs when both arrays are full of NaNs
            return True

        arr1 = arr1[idx]
        arr2 = arr2[idx]
    
    # get the scale of the first array
    scale = np.linalg.norm(arr1.flat, np.inf)
    
    # try to compare the arrays
    with np.errstate(invalid='raise'):
        try:
            is_close = np.any(np.abs(arr1 - arr2) <= (atol + rtol * scale))
        except FloatingPointError:
            is_close = False
        
    return is_close



def is_pos_semidef(x):
    """ checks whether the correlation matrix is positive semi-definite """
    return np.all(np.linalg.eigvals(x) >= 0)



def trim_nan(data, left=True, right=True):
    """ removes nan values from the either end of the array `data`.
    `left` and `right` determine whether these ends of the array are processed.
    The default is to process both ends.
    If data has more than one dimension, the reduction is done along the first
    dimension if any of entry along the other dimension is nan.
    """
    if left:
        # trim left side
        for s in xrange(len(data)):
            if not np.any(np.isnan(data[s])):
                break
    else:
        # don't trim the left side
        s = 0
        
    if right:
        # trim right side
        for e in xrange(len(data) - 1, s, -1):
            if not np.any(np.isnan(data[e])):
                # trim right side
                return data[s:e + 1]
        # array is all nan
        return []
    
    else:
        # don't trim the right side
        return data[s:]
    


def mean(values, empty=0):
    """ calculates mean of generator or iterator.
    Returns `empty` in case of an empty sequence """
    n, total = 0, 0
    for value in values:
        total += value
        n += 1
    return total/n if n > 0 else empty



def moving_average(data, window=1):
    """ calculates a moving average with a given window along the first axis
    of the given data.
    """
    height = len(data)
    result = np.zeros_like(data) + np.nan
    size = 2*window + 1
    assert height >= size
    for pos in xrange(height):
        # determine the window
        if pos < window:
            rows = slice(0, size)
        elif pos > height - window:
            rows = slice(height - size, height)
        else:
            rows = slice(pos - window, pos + window + 1)
            
        # find indices where all values are valid
        cols = np.all(np.isfinite(data[rows, :]), axis=0)
        result[pos, cols] = data[rows, cols].mean(axis=0)
    return result
            



class Interpolate_1D_Extrapolated(interpolate.interp1d):
    """ extend the interpolate class from scipy to return boundary values for
    values beyond the support.
    Here, we return the value at the boundary for all points beyond it.
    """
    
    def __call__(self, x, dtype=np.double):
        """ call the interpolator with appropriate bounds """
        if isinstance(x, np.ndarray):
            # x is a numpy array for which we can have vectorized results
            
            # determine indices
            i_left = (x <= self.x[0])
            i_right = (x >= self.x[-1])
            i_mid = ~i_left & ~i_right #< x is in the right range
            
            y = np.empty_like(x, dtype=dtype)
            y[i_left] = self.y[0]
            y[i_right] = self.y[-1]
            parent = super(Interpolate_1D_Extrapolated, self)
            y[i_mid] = parent.__call__(x[i_mid])
            return y
            
        else:
            # x is simple scalar
            if x < self.x[0]:
                return self.y[0]
            elif x > self.x[-1]:
                return self.y[-1]
            else:
                return super(Interpolate_1D_Extrapolated, self).__call__(x)
            
            

def round_to_even(value):
    """ rounds the value to the nearest even integer """
    return 2*int(value/2 + 0.5)



def round_to_odd(value):
    """ rounds the value to the nearest odd integer """
    return 2*int(value/2) + 1



def get_number_range(dtype):
    """
    determines the minimal and maximal value a certain number type can hold
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
    else:
        raise ValueError('Unsupported data type `%r`' % dtype)

    return info.min, info.max
        

    
def homogenize_arraylist(data):
    """ stores a list of arrays of different length in a single array.
    This is achieved by appending np.nan as necessary.
    """
    len_max = max(len(d) for d in data)
    result = np.empty((len(data), len_max) + data[0].shape[1:], dtype=data[0].dtype)
    result.fill(np.nan)
    for k, d in enumerate(data):
        result[k, :len(d), ...] = d
    return result



def is_equidistant(data):
    """ checks whether the 1d array given by `data` is equidistant """
    if len(data) < 2:
        return True
    diff = np.diff(data)
    return np.allclose(diff, diff.mean())


    
def contiguous_true_regions(condition):
    """ Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index
    Taken from http://stackoverflow.com/a/4495197/932593
    """
    if len(condition) == 0:
        return []
    
    # convert condition array to integer
    condition = np.asarray(condition, np.int)
    
    # Find the indices of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx



def contiguous_int_regions_iter(data):
    """ Finds contiguous regions of the integer array "data". Regions that
    have falsey (0 or False) values are not returned.
    Returns three values (value, start, end), denoting the value and the pairs
    of indices and indicating the start index and end index of the region.
    """
    data = np.asarray(data, int)
    
    # Find the indices of changes in "data"
    d = np.diff(data)
    idx, = d.nonzero() 

    last_k = 0
    for k in idx:
        yield data[k], last_k, k + 1
        last_k = k + 1

    # yield last data point
    if len(data) > 0:
        yield data[-1], last_k, data.size



def safe_typecast(data, dtype):
    """
    truncates the data such that it fits within the supplied dtype.
    This function only supports integer data types so far.
    """
    info = np.iinfo(dtype)
    return np.clip(data, info.min, info.max).astype(dtype)
    
        

class StatisticsAccumulator(object):
    """ class that can be used to calculate statistics of sets of arbitrarily
    shaped data sets. This uses an online algorithm, allowing the data to be
    added one after another """
    
    
    def __init__(self, ddof=0, shape=None, dtype=np.double, ret_cov=False):
        """ initialize the accumulator
        `ddof` is the  delta degrees of freedom, which is used in the formula 
            for the standard deviation.
        `shape` is the shape of the data. If omitted it will be read from the
            first value
        `dtype` is the numpy dtype of the internal accumulator
        `ret_cov` determines whether the covariances are also calculated
        """ 
        self.count = 0
        self.ddof = ddof
        self.dtype = dtype
        self.ret_cov = ret_cov
        
        if shape is None:
            self.shape = None
            self._mean = None
            self._M2 = None
        else:
            self.shape = shape
            size = np.prod(shape)
            self._mean = np.zeros(size, dtype=dtype)
            if ret_cov:
                self._M2 = np.zeros((size, size), dtype=dtype)
            else:
                self._M2 = np.zeros(size, dtype=dtype)
            
            
    @property
    def mean(self):
        """ return the mean """
        if self.shape is None:
            return self._mean
        else:
            return self._mean.reshape(self.shape)
        
        
    @property
    def cov(self):
        """ return the variance """
        if self.count <= self.ddof:
            raise RuntimeError('Too few items to calculate variance')
        elif not self.ret_cov:
            raise ValueError('The covariance matrix was not calculated')
        else:
            return self._M2 / (self.count - self.ddof)
        
        
    @property
    def var(self):
        """ return the variance """
        if self.count <= self.ddof:
            raise RuntimeError('Too few items to calculate variance')
        elif self.ret_cov:
            var = np.diag(self._M2) / (self.count - self.ddof)
        else:
            var = self._M2 / (self.count - self.ddof)

        if self.shape is None:
            return var
        else:
            return var.reshape(self.shape)
        
        
    @property
    def std(self):
        """ return the standard deviation """
        return np.sqrt(self.var)
            
            
    def _initialize(self, value):
        """ initialize the internal state with a given value """
        # state needs to be initialized
        self.count = 1
        if hasattr(value, '__iter__'):
            # make sure the value is a numpy array
            value_arr = np.asarray(value, self.dtype)
            self.shape = value_arr.shape
            size = value_arr.size

            # store 1d version of it
            self._mean = value.flatten()
            if self.ret_cov:
                self._M2 = np.zeros((size, size), self.dtype)
            else:
                self._M2 = np.zeros(size, self.dtype)
                
        else:
            # simple scalar value
            self.shape = None
            self._mean = value
            self._M2 = 0
        
            
    def add(self, value):
        """ add a value to the accumulator """
        if self._mean is None:
            self._initialize(value)
            
        else:
            # update internal state
            if self.shape is not None:
                value = np.ravel(value)
            
            self.count += 1
            delta = value - self._mean
            self._mean += delta / self.count
            if self.ret_cov:
                self._M2 += ((self.count - 1) * np.outer(delta, delta)
                            - self._M2 / self.count)
            else:
                self._M2 += delta * (value - self._mean)
            
    
    def add_many(self, iterator):
        """ adds many values from an array or an iterator """
        for value in iterator:
            self.add(value)
                    
        
        