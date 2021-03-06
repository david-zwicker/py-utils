'''
Created on Feb 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains math functions
'''

from __future__ import division

from collections import Counter
import math

try:
    from math import gcd
except ImportError:
    from fractions import gcd

import numpy as np
from scipy import interpolate
from six.moves import range

from ..misc import estimate_computation_speed


__all__ = ['xlog2x', 'unit_vector', 'heaviside', 'average_angles', 'euler_phi',
           'arrays_close', 'logspace', 'logspace_int', 'is_pos_semidef',
           'trim_nan', 'diff1d_periodic', 'mean', 'moving_average',
           'Interpolate_1D_Extrapolated', 'round_to_even', 'round_to_odd',
           'get_fastest_entropy_function', 'calc_entropy', 'popcount',
           'to_array', 'take_popcount', 'get_number_range',
           'homogenize_arraylist', 'homogenize_unit_array', 'is_equidistant',
           'contiguous_true_regions', 'contiguous_int_regions_iter',
           'safe_typecast', 'range_alternating', 'SmoothData1D']


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



def unit_vector(dim, index):
    """ create a unit vector with a single one and dim - 1 zeros 
    
    Args:
        dim (int): The length of the unit vector, i.e., the space dimension
        index (int): The index which is set to one
        
    Returns:
        numpy.ndarray: the unit vector
    """
    arr = np.zeros(dim)
    arr[index] = 1
    return arr



def heaviside(x, out=None):
    """ defines the Heaviside step function
    
    This uses the half-maximum convention, where heaviside(0) == 0.5
    """
    if out is not None:
        out[:] = 0.5 * (np.sign(x) + 1)
    else:
        return 0.5 * (np.sign(x) + 1)



def average_angles(data, period=PI2):
    """ averages a list of cyclic values (angles by default)
    
    `data` is the list of angles
    `period` denotes the period of the angles (default: 2*pi)
    """
    data = np.asarray(data)
    if period is not PI2:
        data *= PI2 / period
    data = math.atan2(np.sin(data).sum(), np.cos(data).sum())
    if period is not PI2:
        data *= period / PI2
    return data



def euler_phi(n):
    """ evaluates the Euler phi function for argument `n`
    See http://en.wikipedia.org/wiki/Euler%27s_totient_function
    Implementation based on http://stackoverflow.com/a/18114286/932593
    """
    amount = 0

    for k in range(1, n + 1):
        if gcd(n, k) == 1:
            amount += 1

    return amount



def arrays_close(arr1, arr2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """ compares two arrays using a relative and an absolute tolerance
    
    `equal_nan` determines whether two nan values are considered equal or not 
    """
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
    
    # check whether any array has a nan-value 
    if np.any(np.isnan(arr1)) or np.any(np.isnan(arr2)):
        return False
    
    # handle infinities separately
    idx_inf = (np.isinf(arr1) | np.isinf(arr2))
    if np.any(arr1[idx_inf] != arr2[idx_inf]):
        # the arrays differed in a place where at least one was infinite
        return False

    # handle the rest of the entries
    arr1, arr2 = arr1[~idx_inf], arr2[~idx_inf]
    # get the scale of the arrays
    scale = 0.5*(np.abs(arr1) + np.abs(arr2))
    
    # try to compare the arrays
    with np.errstate(invalid='raise'):
        try:
            is_close = np.all(np.abs(arr1 - arr2) <= (atol + rtol * scale))
        except FloatingPointError:
            is_close = False
        
    return is_close




def logspace(start, end, num=None, **kwargs):
    """ Returns an ordered sequence of `num` numbers from `start` to `end`,
    which are spaced logarithmically """
    if start < 0 or end < 0:
        raise ValueError('`start` and `end` must be positive numbers')
    if num is None:
        return np.logspace(np.log10(start), np.log10(end), **kwargs)
    else:
        return np.logspace(np.log10(start), np.log10(end), num=num, **kwargs)



def logspace_int(start, end, num=50):
    """ Returns an ordered sequence of at most `num` integers from `start` to
    `end`, which are approximately spaced logarithmically. Here, `end` is
    included in the sequence. """
    # check whether the supplied number is valid
    num = int(num)
    if num < 0:
        raise ValueError('Number of samples, %s, must be non-negative.' % num)
    if num == 0:
        return np.array([])
    
    # check corner cases
    start = int(start)
    end = int(end)
    if start < 0 or end < 0:
        raise ValueError('`start` and `end` must be positive numbers')
    if num == 1 or start == end:
        return np.array([start])
    
    if start > end:
        # inverted sequence
        return logspace_int(end, start, num)[::-1]
    
    if num > end - start:
        # all integers need to be returned
        return np.arange(start, end + 1)
    
    # calculate the maximal size of underlying logarithmic range
    if start == 0:
        start = 1
        num -= 1
        add_zero = True
    else:
        add_zero = False
        
    num_max = int(np.ceil((math.log(end) - math.log(start)) / 
                          (math.log(end) - math.log(end - 1))))
    a, b = num, num_max  # interval of log-range
    n = a
     
    # try different log-ranges
    while True:
        # determine discretized logarithmic range
        ys = np.logspace(np.log10(start), np.log10(end), num=n)
        ys = np.unique(ys.astype(np.int))
        ys_len = len(ys)

        if ys_len == num:
            break  # reached correct number
            
        elif ys_len < num:
            # n is too small
            a = n
            n = int(math.sqrt(n * b))
            if a == n:
                n += 1
                if n == b:
                    break
            
        elif ys_len > num:
            # n is too large
            b = n
            n = int(math.sqrt(a * n))
            if b == n:
                n -= 1
                if n == a:
                    break

    if add_zero:
        return np.r_[0, ys]
    else:
        return ys



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
        for start, value in enumerate(data):
            if not np.any(np.isnan(value)):
                # found a value that did not contain any nan
                break
        else:
            # array is all nan
            return []
    else:
        # don't trim the left side
        start = 0
        
    if right:
        # trim right side
        for end in range(len(data) - 1, start - 1, -1):
            if not np.any(np.isnan(data[end])):
                # found a value that did not contain any nan
                return data[start:end + 1]
        # array is all nan
        return []
    
    else:
        # don't trim the right side
        return data[start:]
    


def diff1d_periodic(data, period):
    """ calculates the discrete difference for a circular integer variable with
    given period """
    data = np.asanyarray(data)
    dtype = data.dtype
    
    if np.issubdtype(dtype, np.unsignedinteger):
        diff = np.diff(np.r_[data.astype(np.int), data[0]])
        return (diff + period//2) % period - period//2
    
    elif np.issubdtype(dtype, np.integer):
        diff = np.diff(np.r_[data, data[0]])
        return (diff + period//2) % period - period//2
    
    elif np.issubdtype(dtype, np.number):
        diff = np.diff(np.r_[data, data[0]])
        return (diff + period/2) % period - period/2
    
    else:
        raise TypeError('Cannot calculate the difference for type `%s`' % dtype)



def mean(values, empty=0):
    """ calculates mean of generator or iterator.
    Returns `empty` in case of an empty sequence """
    n, mean = 0, 0
    for value in values:
        n += 1
        mean += (value - mean) / n
    return mean if n > 0 else empty



def moving_average(data, window=1):
    """ calculates a moving average with a given window along the first axis
    of the given data.
    Inspired by http://stackoverflow.com/a/14314054/932593
    """
    ret = np.cumsum(data, dtype=float, axis=0)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window
            


class Interpolate_1D_Extrapolated(interpolate.interp1d):
    """ extend the interpolate class from scipy to return boundary values for
    values beyond the support.
    Here, we return the value at the boundary for all points beyond it.
    """
    
    def __call__(self, x, dtype=np.double):
        """ call the interpolator with appropriate bounds """
        if hasattr(x, '__iter__'):
            # interpret x as a numpy array
            x = np.asarray(x)
            
            # determine indices
            i_left = (x <= self.x[0])
            i_right = (x >= self.x[-1])
            i_mid = ~i_left & ~i_right  # x is in the right range
            
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



def _entropy_numpy(arr):
    """
    calculate the base 2 entropy of the distribution given in `arr` using the
    numpy.unique function
    """
    counts = np.unique(arr, return_counts=True)[1]
    fs = np.true_divide(counts, len(arr))
    return -np.sum(fs * np.log2(fs))


def _entropy_counter1(arr):
    """
    calculate the base 2 entropy of the distribution given in `arr` using a
    `Counter` and the `itervalues` method (for python2)
    """
    arr_len = len(arr)
    if arr_len == 0:
        return 0
    log_arr_len = np.log2(len(arr))
    return -sum(val * (np.log2(val) - log_arr_len)
                for val in Counter(arr).itervalues()) / arr_len

    
def _entropy_counter2(arr):
    """
    calculate the base 2 entropy of the distribution given in `arr` using a
    `Counter` and the `values` method (for python3)
    """
    arr_len = len(arr)
    if arr_len == 0:
        return 0
    log_arr_len = np.log2(len(arr))
    return -sum(val * (np.log2(val) - log_arr_len)
                for val in Counter(arr).values()) / arr_len


# compile the list of functions that can calculate an entropy
_ENTROPY_FUNCTIONS = [_entropy_numpy, _entropy_counter1, _entropy_counter2]


def get_fastest_entropy_function():
    """ returns a function that calculates the base 2 entropy of a array of
    integers. Here, several alternative definitions are tested and the fastest
    one is returned """
    # test all functions against a random array to find the fastest one
    test_array = np.random.randint(0, 10, 100)
    func_fastest, speed_max = None, 0
    for test_func in _ENTROPY_FUNCTIONS:
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
    """ calculate the base 2 entropy of the array given by `arr`.
    The first time this function is called, it runs a benchmark to determine the
    fastest method to calculate the entropy. All subsequent calls then use this
    method.
    """
    try:
        return calc_entropy._function(arr)
    except AttributeError:
        calc_entropy._function = get_fastest_entropy_function()
        return calc_entropy._function(arr)



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
    x += x >> 8   # put count of each 16 bits into their lowest 8 bits
    x += x >> 16  # put count of each 32 bits into their lowest 8 bits
    x += x >> 32  # put count of each 64 bits into their lowest 8 bits
    return x & 0x7f



def to_array(iter_or_list):
    """ turns the given item into a numpy array """
    if isinstance(iter_or_list, np.ndarray):
        # input is already a numpy array
        return iter_or_list 

    try:
        # input is an iterator that also supports the __len__ method
        return np.fromiter(iter_or_list, len(iter_or_list))
    except TypeError:
        # otherwise we just turn it into a list first
        return np.array(list(iter_or_list))
    
    
        
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
        raise TypeError('Unsupported data type `%r`' % dtype)

    return info.min, info.max
        

    
def homogenize_arraylist(data):
    """ stores a list of arrays of different length in a single array.
    This is achieved by appending np.nan as necessary.
    """
    if len(data) == 0:
        return data

    # try getting the lengths of the items
    try:    
        len_max = max(len(d) for d in data)
    except TypeError:
        raise TypeError('Expected list of lists/arrays.')
    
    data0 = np.asarray(data[0])
    
    result = np.empty((len(data), len_max) + data0.shape[1:],
                      dtype=data0.dtype)
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
        


def is_equidistant(data, **kwargs):
    """ checks whether the 1d array given by `data` is equidistant """
    if len(data) < 2:
        return True
    diff = np.diff(data)
    return np.allclose(diff, diff.mean(), **kwargs)


    
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
    idx = np.flatnonzero(d)

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
    return idx.reshape(-1, 2)



def contiguous_int_regions_iter(data):
    """ Finds contiguous regions of the integer array "data". 
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
    This function only supports integer data types.
    This function fails if the magnitude of the input is above
    18446744073709551615, which is np.iinfo(np.uint64).max
    """
    info = np.iinfo(dtype)
    return dtype(np.clip(data, info.min, info.max))
    


def range_alternating(stop, step=1):
    """ returns a generator that iterates over alternating numbers starting at 0
    until `stop` is reached. The step size can be given by `step`.
    
    Example:
        >>> list(range_alternating(4, 2))
        [0, 2, -2, 4, -4]
    """
    if step == 0:
        raise ValueError('range_alternating() step argument must not be zero')
    if stop == 0:
        return
    
    i = 0
    while True:
        yield i
        i = -i + step
        if abs(i) >= abs(stop):
            return
        yield i
        i *= -1
    yield i
    
    
    
class SmoothData1D():
    """ provide structure for smoothing data in 1d """
    
    sigma_auto_scale = 10  # scale for determining automatic values for sigma
    
    def __init__(self, x, y, sigma=None):
        """ initialize with data
        
        `sigma` determines the smoothing window using units of `x`
        """
        self.x = np.ravel(x)
        self.y = np.ravel(y)
        if self.x.shape != self.y.shape:
            raise ValueError('`x` and `y` must have equal number of elements')
            
        if sigma is None:
            sigma = self.sigma_auto_scale * self.x.ptp() / len(self.x)
        self._scale = 0.5 * sigma**-2
        
        
    @property
    def bounds(self):
        """ return minimal and maximal `x` values """
        return self.x.min(), self.x.max()
        
        
    def __call__(self, xs):
        """ return smoothed y values for the positions given in `xs` """
        xs = np.asanyarray(xs)
        shape = xs.shape
        xs = np.ravel(xs)
        with np.errstate(under='ignore'):
            weight = np.exp(-self._scale * (self.x[:, None] - xs[None, :])**2)
            weight /= weight.sum(axis=0)
        result = np.dot(self.y, weight)
        return result.reshape(shape)    
    
    