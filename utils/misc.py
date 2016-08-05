'''
Created on Jan 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import contextlib
import functools
import itertools
import logging
import sys
import timeit
import types
import warnings
from collections import Counter

import numpy as np
from scipy.stats import itemfreq


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    logging.warn('Package tqdm could not be imported and progress bars are '
                 'thus not available.')



def score_interaction_matrices(I1, I2):
    """ returns a score of the similarity of the interaction matrices, taking
    into account all permutations of the receptors """
    assert I1.shape == I2.shape
    
    return min(np.abs(I1[perm, :] - I2).mean()
               for perm in itertools.permutations(range(len(I1))))



def estimate_computation_speed(func, *args, **kwargs):
    """ estimates the computation speed of a function """
    test_duration = kwargs.pop('test_duration', 1)
    
    # prepare the function
    if args or kwargs:
        test_func = functools.partial(func, *args, **kwargs)
    else:
        test_func = func
    
    # call function once to allow caches be filled
    test_func()
     
    # call the function until the total time is achieved
    number, duration = 1, 0
    while duration < 0.1*test_duration:
        number *= 10
        duration = timeit.timeit(test_func, number=number)
    return number/duration



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



class classproperty(object):
    """ decorator that can be used to define read-only properties for classes. 
    Code copied from http://stackoverflow.com/a/5192374/932593
    """
    def __init__(self, f):
        self.f = f
        
    def __get__(self, obj, owner):
        return self.f(owner)
    
    
    
class DummyFile(object):
    """ dummy file that ignores all write calls """
    def write(self, x):
        pass



@contextlib.contextmanager
def silent_stdout():
    """
    context manager that silence the standard output
    Code copied from http://stackoverflow.com/a/2829036/932593
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
    
    

if sys.version_info[0] == 2:
    # python 2 version
    def copy_func(f, name=None):
        """ copies a python function. Taken from
        http://stackoverflow.com/a/6528148/932593
        """ 
        return types.FunctionType(f.func_code, f.func_globals,
                                  name or f.func_name, f.func_defaults,
                                  f.func_closure)

else:
    # python 3 version as the default for compatibility
    def copy_func(f, name=None):
        """ copies a python function. Inspired from
        http://stackoverflow.com/a/6528148/932593
        """ 
        return types.FunctionType(f.__code__, f.__globals__, name or f.__name__,
                                  f.__defaults__, f.__closure__)



class DeprecationHelper(object):
    """
    Helper function for re-routing deprecated classes 
    copied from http://stackoverflow.com/a/9008509/932593
    """
    
    def __init__(self, new_target, warning_class=Warning):
        self.new_target = new_target
        self.warning_class = warning_class

    def _warn(self):
        msg = "The class was renamed to `%s`"  % self.new_target.__name__
        warnings.warn(msg, self.warning_class, stacklevel=3)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self.new_target, attr)
    
    

class CachedArray(object):
    def __init__(self, value=None):
        self._data = np.empty(0)
        self.value = value
    
    def __call__(self, shape):
        if self._data.shape == shape:
            if self.value is not None:
                self._data.fill(self.value)
        else:
            if self.value is None:
                self._data = np.empty(shape)
            elif self.value == 0:
                self._data = np.zeros(shape)
            else: 
                self._data = np.full(shape, self.value, np.double)
        return self._data



            
            

def display_progress(iterator, total=None, mininterval=5):
    """
    displays a progress bar when iterating
    """
    if tqdm is not None:
        return tqdm(iterator, total=total, leave=True, mininterval=mininterval)
    else:
        return iterator
    
    
    
def get_loglevel_from_name(name_or_int):
    """ converts a logging level name to the numeric representation """
    # see whether it is already an integer
    if isinstance(name_or_int, int):
        return name_or_int
    
    # convert it from the name
    level = logging.getLevelName(name_or_int.upper())
    if isinstance(level, int):
        return level
    else:
        raise ValueError('`%s` is not a valid logging level.' % name_or_int)



def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    Taken from http://code.activestate.com/recipes/391367-deprecated/
    """
    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc



def unique_based_on_id(data):
    """ returns a list with only unique items, where the uniqueness
    is determined from the id of the items. This can be useful in the
    case where the items cannot be hashed and a set can thus not be used. """
    result, seen = [], set()
    for item in data:
        if id(item) not in seen:
            result.append(item)
            seen.add(id(item))
    return result
