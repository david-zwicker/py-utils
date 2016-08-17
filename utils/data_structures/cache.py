'''
Created on Sep 11, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains functions that can be used to manage cache structures
'''

from __future__ import division

import collections
import functools
import types

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np



class DictFiniteCapacity(collections.OrderedDict):
    """ cache with a limited number of items """
    
    default_capacity = 100
    
    def __init__(self, *args, **kwargs):
        self.capacity = kwargs.pop('capacity', self.default_capacity)
        super(DictFiniteCapacity, self).__init__(*args, **kwargs)


    def check_length(self):
        """ ensures that the dictionary does not grow beyond its capacity """
        while len(self) > self.capacity:
            self.popitem(last=False)
            

    def __setitem__(self, key, value):
        super(DictFiniteCapacity, self).__setitem__(key, value)
        self.check_length()
        
        
    def update(self, values):
        super(DictFiniteCapacity, self).update(values)
        self.check_length()
        
    

class cached_property(object):
    """Decorator to use a function as a cached property.

    The function is only called the first time and each successive call returns
    the cached result of the first call.

        class Foo(object):

            @cached_property
            def foo(self):
                return "Cached"
                
    The data is stored in a dictionary named `_cache` attached to the instance
    of each object. The cache can thus be cleared by setting self._cache = {}

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.
    """

    def __init__(self, *args, **kwargs):
        """ setup the decorator """
        self.cache = None
        if len(args) > 0:
            if callable(args[0]):
                # called as a plain decorator
                self.__call__(*args, **kwargs)
            else:
                # called with arguments
                self.cache = args[0]
        else:
            # called with arguments
            self.cache = kwargs.pop('cache', self.cache)
            

    def __call__(self, func, doc=None, name=None):
        """ save the function to decorate """
        self.func = func
        self.__doc__ = doc or func.__doc__
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        return self


    def __get__(self, obj, owner):
        if obj is None:
            return self

        # load the cache structure
        if self.cache is None:
            try:
                cache = obj._cache
            except AttributeError:
                cache = obj._cache = {}
        else:
            try:
                cache = getattr(obj, self.cache)
            except:
                cache = {}
                setattr(obj, self.cache, cache)
                
        # try to retrieve from cache or call and store result in cache
        try:
            value = cache[self.__name__]
        except KeyError:
            value = self.func(obj)
            cache[self.__name__] = value
        return value
    


class PersistentDict(collections.MutableMapping):
    """ a key value database which is stored on the disk
    keys and values must be strings.
    """
    
    def __init__(self, filename):
        # lazy import
        import sqlite3
        # open the sqlite table
        self._con = sqlite3.connect(filename)
        # make sure that the cache table exists
        with self._con:
            self._con.execute("CREATE table IF NOT EXISTS cache ("
                                  "key TEXT PRIMARY KEY, "
                                  "value TEXT"
                              ");")
        
        
    def __del__(self):
        self._con.close()
        
        
    def __len__(self):
        return self._con.execute("SELECT Count(*) FROM cache").fetchone()[0]
    
    
    def __getitem__(self, key):
        res = self._con.execute("SELECT value FROM cache WHERE key=? "
                                "LIMIT 1", (key,)).fetchone()
        if res:
            return res[0]
        else:
            raise KeyError(key)
        
        
    def __setitem__(self, key, value):
        with self._con:
            self._con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)",
                              (key, value))


    def __delitem__(self, key):
        with self._con:
            self._con.execute("DELETE FROM cache where key=?", (key,))
    
    
    def __contains__(self, key):
        return self._con.execute("SELECT EXISTS(SELECT 1 FROM cache "
                                 "WHERE key=? LIMIT 1);", (key,)).fetchone()[0]
    
    
    def __iter__(self):
        for row in self._con.execute("SELECT key FROM cache").fetchall():
            yield row[0]



class PersistentSerializedDict(PersistentDict):
    """ a key value database which is stored on the disk
    This class provides hooks for converting arbitrary keys and values to
    strings, which are then stored in the database. If not overwritten, pickled
    strings are used by default
    """
    
    def __init__(self, filename, serialize_keys=True, serialize_values=True):
        super(PersistentSerializedDict, self).__init__(filename)
        self.pickle_keys = serialize_keys
        self.pickle_values = serialize_values
    
    
    def key_to_str(self, key):
        """ converts an arbitrary key to a string """
        return pickle.dumps(key) if self.pickle_keys else key
    
    
    def value_to_str(self, value):
        """ converts an arbitrary value to a string """
        return pickle.dumps(value) if self.pickle_values else value
    
    
    def str_to_value(self, value):
        """ converts a string into a value object """
        return pickle.loads(str(value)) if self.pickle_values else value
    
    
    def __getitem__(self, key):
        # convert key to its string representation
        key = self.key_to_str(key)
        # fetch the value
        value = super(PersistentSerializedDict, self).__getitem__(key)
        # convert the value to its object representation
        return self.str_to_value(value)
        
        
    def __setitem__(self, key, value):
        # convert key and value to their string representations
        key = self.key_to_str(key)
        value = self.value_to_str(value)
        super(PersistentSerializedDict, self).__setitem__(key, value)


    def __delitem__(self, key):
        # convert key to its string representation
        key = self.key_to_str(key)
        super(PersistentSerializedDict, self).__delitem__(key)
    
    
    def __contains__(self, key):
        # convert key to its string representation
        key = self.key_to_str(key)
        super(PersistentSerializedDict, self).__contains__(key)
    
    
    def __iter__(self):
        for value in super(PersistentSerializedDict, self).__iter__():
            # convert the value to its object representation
            yield self.str_to_value(value)



def cached_method(method, doc=None, name=None):
    """ decorator that caches method calls in a dictionary attached to the
    object. This can be used with most classes

        class Foo(object):

            @cached_method
            def foo(self):
                return "Cached"
    
            @cached_method
            def bar(self):
                return "Cached"
                
    
        foo = Foo()
        foo.bar()
        
        # The cache is now stored in foo._cache
        
    This class also plays together with user-supplied storage backends, if the
    method `get_cache` is defined.  
    
        class Foo(object):
        
            def __init__(self):
                self._cache = {}
                
            def get_cache(self, name):
                try:
                    return self._cache[name]
                except:
                    cache = {}
                    self._cache[name] = cache
                    return cache

            @cached_method
            def foo(self):
                return "Cached"
    """

    if name is None:
        name = method.__name__

    def get_cache_method(obj, name):
        """ universial method that returns a dict cache for name `name` """
        try:
            return obj._cache[name]
        except AttributeError:
            try:
                obj.init_cache(obj)
            except AttributeError:
                obj._cache = collections.defaultdict(dict)
            return obj._cache[name]

    def make_cache_key_method(args, kwargs):
        """ universial method that converts methods arguments to a string """
        return pickle.dumps((args, kwargs))


    @functools.wraps(method)
    def wrapper(obj, *args, **kwargs):
        try:
            # try loading the cache_getter from the object
            get_cache = obj.get_cache
        except AttributeError:
            # otherwise use the default method from DictCache
            obj.get_cache = types.MethodType(get_cache_method, obj.__class__)
            get_cache = obj.get_cache

        try:
            # try loading the cache_getter from the object
            make_cache_key = obj.make_cache_key
        except AttributeError:
            # otherwise use the default method from DictCache
            obj.make_cache_key = make_cache_key_method
            make_cache_key = obj.make_cache_key

        # obtain the actual cache associated with this method
        cache = get_cache(name)
        # determine the key that encodes the current arguments
        cache_key = make_cache_key(args, kwargs)

        try:
            # try loading the results from the cache
            result = cache[cache_key]
        except KeyError:
            # if this failed, compute and store the results
            result = method(obj, *args, **kwargs)
            cache[cache_key] = result
        return result

    return wrapper



class CachedArray(object):
    """
    class that provides an array of given shape when called. If the shape is
    consistent with the last call, a stored copy will be returned. Otherwise a
    new array will be constructed.
    """
    
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
