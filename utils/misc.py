'''
Created on Jan 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division, absolute_import

import functools
import logging
import itertools
import importlib
import math
import sys
import re
import timeit
import types
import warnings

from six.moves import zip

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None



def estimate_computation_speed(func, *args, **kwargs):
    """ estimates the computation speed of a function
    
    Returns the number of times the function can be calculated in one second.
    The inverse is thus the runtime in seconds per function call
    """
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
    while duration < 0.1 * test_duration:
        number *= 10
        duration = timeit.timeit(test_func, number=number)
    return number / duration



class classproperty(object):
    """ decorator that can be used to define read-only properties for classes. 
    Code copied from http://stackoverflow.com/a/5192374/932593
    """
    def __init__(self, f):
        self.f = f
        
    def __get__(self, obj, owner):
        return self.f(owner)



class hybridmethod:
    """
    descriptor that can be used as a decorator to allow calling a method both
    as a classmethod and an instance method
     
    Adapted from https://stackoverflow.com/a/28238047
    """
    
    def __init__(self, fclass, finstance=None, doc=None):
        self.fclass = fclass
        self.finstance = finstance
        self.__doc__ = doc or fclass.__doc__
        # support use on abstract base classes
        self.__isabstractmethod__ = bool(
            getattr(fclass, '__isabstractmethod__', False)
        )
        

    def classmethod(self, fclass):
        return type(self)(fclass, self.finstance, None)


    def instancemethod(self, finstance):
        return type(self)(self.fclass, finstance, self.__doc__)


    def __get__(self, instance, cls):
        if instance is None or self.finstance is None:
            # either bound to the class, or no instance method available
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)    
    
    
    
class DummyFile(object):
    """ dummy file that ignores all write calls """
    def write(self, x):
        pass
    
    def flush(self):
        pass



class RedirectedStdout(object):
    """
    context manager that redirects the standard output to the given stream
    """

    def __init__(self, stream):
        self._target = stream
        self._saved_stdout = None
    
    def __enter__(self):
        self._saved_stdout = sys.stdout
        sys.stdout = self._target
        
    def __exit__(self, type, value, traceback):  # @ReservedAssignment
        sys.stdout = self._saved_stdout
        


def silent_stdout():
    """ context manager that silence the standard output """
    return RedirectedStdout(DummyFile())



class RedirectedStderr(object):
    """
    context manager that redirects the standard error stream to the given stream
    """

    def __init__(self, stream):
        self._target = stream
        self._saved_stderr = None
    
    def __enter__(self):
        self._saved_stdoerr = sys.stderr
        sys.stderr = self._target
        
    def __exit__(self, type, value, traceback):  # @ReservedAssignment
        sys.stderr = self._saved_stderr
        


def silent_stderr():
    """ context manager that silence the standard error stream """
    return RedirectedStderr(DummyFile())
    
    
    
class DisableLogging(object):
    """ class that temporarily disables all logging """
    
    def __init__(self):
        """ initialize the class properties """
        self._root_logger = None
        self._state = None
    
    
    def __enter__(self):
        """ store the state of the root logger and disable it """
        if self._root_logger is None:
            self._root_logger = logging.getLogger()
        self._state = self._root_logger.disabled
        self._root_logger.disabled = True


    def __exit__(self, *args):
        """ restore the state of the root logger """
        self._root_logger.disabled = self._state
    
    

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
    
    Use as `OldClsName = DeprecationHelper(NewClsName)`
    """
    
    def __init__(self, new_target, warning_class=DeprecationWarning):
        """ initialize the decorator """
        self.new_target = new_target
        self.warning_class = warning_class

    def _warn(self):
        """ emit a warning that the class was renamed """
        msg = "The class was renamed to `%s`" % self.new_target.__name__
        warnings.warn(msg, self.warning_class, stacklevel=3)

    def __call__(self, *args, **kwargs):
        """ this should catch the normal initialization """
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        """ overwrite this to also capture classmethods """
        self._warn()
        return getattr(self.new_target, attr)
    
    

def display_progress(iterator, total=None, enabled=True, **kwargs):
    """
    displays a progress bar when iterating
    """
    if not enabled:
        return iterator
    
    if tqdm is not None:
        return tqdm(iterator, total=total, **kwargs)
    else:
        if display_progress._show_warning:
            logging.getLogger(__name__).warn('Module `tqdm` is not available '
                                             'and progress cannot be displayed')
            display_progress._show_warning = False
        return iterator

    
display_progress._show_warning = True
    
    
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



def iter_pairwise(iterable, circular=False):
    """ iterates over pairs of the iterable """
    a, b = itertools.tee(iterable)
    first_item = next(b, None)
    if circular:
        b = itertools.chain(b, [first_item])
    return zip(a, b)



def format_timedelta(value, time_format=None):
    """ formats a datetime.timedelta with the given format.
    Code copied from Django as explained in
    http://stackoverflow.com/a/30339105/932593
    """ 
    
    if time_format is None:
        time_format = "{days} days, {hours2}:{minutes2}:{seconds2}"

    if hasattr(value, 'seconds'):
        seconds = value.seconds + value.days * 24 * 3600
    else:
        seconds = int(value)

    seconds_total = seconds

    minutes = int(math.floor(seconds / 60))
    minutes_total = minutes
    seconds -= minutes * 60

    hours = int(math.floor(minutes / 60))
    hours_total = hours
    minutes -= hours * 60

    days = int(math.floor(hours / 24))
    days_total = days
    hours -= days * 24

    years = int(math.floor(days / 365))
    years_total = years
    days -= years * 365

    return time_format.format(**{
        'seconds': seconds,
        'seconds2': str(seconds).zfill(2),
        'minutes': minutes,
        'minutes2': str(minutes).zfill(2),
        'hours': hours,
        'hours2': str(hours).zfill(2),
        'days': days,
        'years': years,
        'seconds_total': seconds_total,
        'minutes_total': minutes_total,
        'hours_total': hours_total,
        'days_total': days_total,
        'years_total': years_total,
    })
    
    
    
def replace_words(text, replacements):
    """ applies the replacement rules given in `replacements` to the text. The
    replacements are given as a simple dictionary.
    
    The function is inspired by https://stackoverflow.com/a/17730939/932593 
    """
    def replace(match):
        return replacements[match.group(0)]

    pattern = '|'.join(r'\b%s\b' % re.escape(s) for s in replacements) 
    return re.sub(pattern, replace, text)



def import_class(identifier):
    """ import a class or module given an identifier 
    
    The identifier can be a module or a class:
        identifier == 'numpy.linalg.norm'
    is equivalent to
        from numpy.linalg import norm
    and would return a reference to `norm`
    """
    module_path, _, class_name = identifier.rpartition('.')
    if module_path:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        # this happens when identifier does not contain a dot
        return importlib.import_module(class_name)
