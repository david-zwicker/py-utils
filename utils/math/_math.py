'''
Created on Feb 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains math functions
'''

from __future__ import division

from collections import Counter
import fractions
import math

import numpy as np
from scipy import interpolate
from scipy.stats import itemfreq

from ..misc import estimate_computation_speed



# constants
PI2 = 2*np.pi



def xlog2x(x):
    """ calculates x*np.log2(x) """
    if x == 0:
        return 0
    else:
        return x * np.log2(x)

# vectorize the function above
xlog2x = np.vectorize(xlog2x, otypes='d')



def average_angles(data, period=PI2):
    """ averages a list of cyclic values (angles by default)
    data  The list of angles
    per   The period of the angular variables
    """
    data = np.asarray(data)    
    if period is not PI2:
        data *= PI2/period
    data = math.atan2(np.sin(data).sum(), np.cos(data).sum())
    if period is not PI2:
        data *= period/PI2
    return data



def euler_phi(n):
    """ evaluates the Euler phi function for argument `n`
    See http://en.wikipedia.org/wiki/Euler%27s_totient_function
    Implementation based on http://stackoverflow.com/a/18114286/932593
    """
    amount = 0

    for k in xrange(1, n + 1):
        if fractions.gcd(n, k) == 1:
            amount += 1

    return amount



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




def logspace(start, end, num=None, dtype=None):
    """ Returns an ordered sequence of `num` numbers from `start` to `end`,
    which are spaced logarithmically """
    if num is None:
        return np.logspace(np.log10(start), np.log10(end), dtype=dtype)
    else:
        return np.logspace(np.log10(start), np.log10(end), num=num, dtype=dtype)



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



def get_fastest_entropy_function():
    """ returns a function that calculates the entropy of a array of integers
    Here, several alternative definitions are tested and the fastest one is
    returned """
    
    # define a bunch of functions that act the same but have different speeds 
    
    def entropy_numpy(arr):
        """ calculate the entropy of the distribution given in `arr` """
        fs = np.unique(arr, return_counts=True)[1]
        return np.sum(fs*np.log2(fs))
    
    def entropy_scipy(arr):
        """ calculate the entropy of the distribution given in `arr` """
        fs = itemfreq(arr)[:, 1]
        return np.sum(fs*np.log2(fs))
    
    def entropy_counter1(arr):
        """ calculate the entropy of the distribution given in `arr` """
        return sum(val*np.log2(val)
                   for val in Counter(arr).itervalues())
        
    def entropy_counter2(arr):
        """ calculate the entropy of the distribution given in `arr` """
        return sum(val*np.log2(val)
                   for val in Counter(arr).values())

    # test all functions against a random array to find the fastest one
    test_array = np.random.random_integers(0, 10, 100)
    func_fastest, speed_max = None, 0
    for test_func in (entropy_numpy, entropy_scipy, entropy_counter1,
                      entropy_counter2):
        try:
            speed = estimate_computation_speed(test_func, test_array)
        except (TypeError, AttributeError):
            # TypeError: older numpy versions don't support `return_counts`
            # AttributeError: python3 does not have iteritems
            pass
        else:
            if speed > speed_max:
                func_fastest, speed_max = test_func, speed

    return func_fastest


def calc_entropy(arr):
    """ calculate the entropy of the distribution given in `arr` """
    # find the fastest entropy function on the first call of this function
    # and bind it to the same name such that it is used in future times
    global calc_entropy
    calc_entropy = get_fastest_entropy_function()
    return calc_entropy(arr)



def popcount(x):
    """
    count the number of high bits in the integer `x`.
    Taken from https://en.wikipedia.org/wiki/Hamming_weight
    """
    # put count of each 2 bits into those 2 bits 
    x -= (x >> 1) & 0x5555555555555555 
    # put count of each 4 bits into those 4 bits
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    # put count of each 8 bits into those 8 bits 
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f  
    x += x >>  8 # put count of each 16 bits into their lowest 8 bits
    x += x >> 16 # put count of each 32 bits into their lowest 8 bits
    x += x >> 32 # put count of each 64 bits into their lowest 8 bits
    return x & 0x7f



def take_popcount(arr, n):
    """ returns only those parts of an array whose indices have a given
    popcount """
    return [v for i, v in enumerate(arr) if popcount(i) == n]


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



def homogenize_unit_array(arr, unit=None):
    """ takes an list with quantities and turns it into a numpy array with a
    single associated unit
    
    `arr` is the (not nested) list of values
    `unit` defines the unit to which all items are converted. If it is omitted,
        the unit is determined automatically from the first array element
    """
    if unit is None:
        try:
            # extract unit from array
            unit = arr[0] / arr[0].magnitude
        except (AttributeError, IndexError):
            # either `arr` was not an array or it didn't carry units 
            return arr
        
    # convert all values to this unit and return their magnitude
    arr = [val.to(unit).magnitude for val in arr]
    
    # return the array with units
    return arr * unit
        


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
    

    