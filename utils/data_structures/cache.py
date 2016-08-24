'''
Created on Sep 11, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains functions that can be used to manage cache structures
'''

from __future__ import division

import collections
import functools
import numpy as np



def make_serializer(method):
    """ returns a function that serialize data with the  given method """
    if method is None or not method:
        return lambda s: s

    elif method == 'json':
        import json
        return lambda s: json.dumps(s, sort_keys=True)

    elif method == 'pickle':
        import cPickle
        return cPickle.dumps

    elif method == 'yaml':
        import yaml
        return yaml.dump
    
    else:
        raise ValueError('Unknown serialization method `%s`' % method)



def make_unserializer(method):
    """ returns a function that unserialize data with the  given method """
    if method is None or not method:
        return lambda s: s

    elif method == 'json':
        import json
        return json.loads

    elif method == 'pickle':
        import cPickle
        return lambda s: cPickle.loads(str(s))

    elif method == 'yaml':
        import yaml
        return yaml.load
    
    else:
        raise ValueError('Unknown serialization method `%s`' % method)
    
    

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
    strings, which are then stored in the database.
    """
    
    def __init__(self, filename, key_serialization='json',
                 value_serialization='pickle'):
        """ initializes a persistent dictionary whose keys and values are
        serialized transparently. The serialization methods are determined by
        `key_serialization` and `value_serialization`.
        """
        super(PersistentSerializedDict, self).__init__(filename)
        
        self.serialize_key = make_serializer(key_serialization)
        self.serialize_value = make_serializer(value_serialization)
        self.unserialize_value = make_unserializer(value_serialization)
    
    
    def __getitem__(self, key):
        # convert key to its string representation
        key = self.serialize_key(key)
        # fetch the value
        value = super(PersistentSerializedDict, self).__getitem__(key)
        # convert the value to its object representation
        return self.unserialize_value(value)
        
        
    def __setitem__(self, key, value):
        # convert key and value to their string representations
        key = self.serialize_key(key)
        value = self.serialize_value(value)
        # add the item to the dictionary
        super(PersistentSerializedDict, self).__setitem__(key, value)


    def __delitem__(self, key):
        # convert key to its string representation
        key = self.serialize_key(key)
        # delete the item from the dictionary
        super(PersistentSerializedDict, self).__delitem__(key)
    
    
    def __contains__(self, key):
        # convert key to its string representation
        key = self.serialize_key(key)
        # check whether this items exists in the dictionary
        super(PersistentSerializedDict, self).__contains__(key)
    
    
    def __iter__(self):
        # iterate  dictionary
        for value in super(PersistentSerializedDict, self).__iter__():
            # convert the value to its object representation
            yield self.unserialize_value(value)



class cached_method(object):
    """ class handling the caching of results of methods """
    
    def __init__(self, factory=None, serializer='json', doc=None, name=None):
        """ decorator that caches method calls in a dictionary attached to the
        methods. This can be used with most classes
    
            class Foo(object):
    
                @cached_method()
                def foo(self):
                    return "Cached"
        
                @cached_method()
                def bar(self):
                    return "Cached"
                    
        
            foo = Foo()
            foo.bar()
            
            # The cache is now stored in foo.foo._cache and foo.bar._cache
            
        This class also plays together with user-supplied storage backends by 
        defining a cache factory
        
            class Foo(object):
                    
                def get_cache(self, name):
                    # `name` is the name of the method to cache 
                    return DictFiniteCapacity()
    
                @cached_method(factory='get_cache')
                def foo(self):
                    return "Cached"
        """
        # check whether the decorator has been applied correctly
        if callable(factory):
            class_name = self.__class__.__name__
            raise ValueError('Missing function call. Call this decorator as '
                             '@{0}() instead of @{0}'.format(class_name))
            
        self.factory = factory
        self.serializer = serializer
        self.name = name
        
    
    def __call__(self, method):
        """ apply the cache decorator """
        
        if self.name is None:
            self.name = method.__name__
    
        serialize_key = make_serializer(self.serializer)
    
        @functools.wraps(method)
        def wrapper(obj, *args, **kwargs):
            # try accessing the cache
            try:
                cache = wrapper._cache
            except AttributeError:
                # cache was not available and we thus need to create it
                if self.factory is None:
                    cache = {}
                else:
                    cache = getattr(obj, self.factory)(self.name)
                    
                # attach the cache to the wrapper
                wrapper._cache = cache
    
            # determine the key that encodes the current arguments
            cache_key = serialize_key((args, kwargs))
    
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
