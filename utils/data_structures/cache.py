'''
Created on Sep 11, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains functions that can be used to manage cache structures
'''

from __future__ import division

import collections
import functools
import os
import sys

import six

import numpy as np



def make_serializer(method):
    """ returns a function that serialize data with the  given method """
    if method is None:
        return lambda s: s

    elif method == 'json':
        import json
        return lambda s: json.dumps(s, sort_keys=True).encode('utf-8')

    elif method == 'pickle':
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        return pickle.dumps

    elif method == 'yaml':
        import yaml
        return lambda s: yaml.dump(s).encode('utf-8')
    
    else:
        raise ValueError('Unknown serialization method `%s`' % method)



def make_unserializer(method):
    """ returns a function that unserialize data with the  given method """
    if method is None:
        return lambda s: s

    elif method == 'json':
        import json
        return lambda s: json.loads(s.decode('utf-8'))

    elif method == 'pickle':
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
            
        if sys.version_info[0] > 2:
            return lambda s: pickle.loads(s)
        else:
            # python 2 sometimes needs an explicit conversion to string
            return lambda s: pickle.loads(str(s))

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
        


class PersistentDict(collections.MutableMapping):
    """ a key value database which is stored on the disk
    keys and values must be strings.
    """
    
    def __init__(self, filename):
        # open the sqlite table
        self.filename = filename
        self.open()
        
    
    def open(self):
        """ opens the database assuming that it is not open """
        # lazy import
        import sqlite3
        self._con = sqlite3.connect(self.filename)
        self._con.text_factory = bytes  # make sure that we mainly handle bytes 
        
        # make sure that the cache table exists
        with self._con:
            self._con.execute("CREATE table IF NOT EXISTS cache ("
                                  "key BLOB PRIMARY KEY, "
                                  "value BLOB"
                              ");")
            
            
    def clear(self):
        """ closes and opens the database """
        self._con.close()
        os.remove(self.filename)
        self.open()
            
        
    def __del__(self):
        self._con.close()
        
        
    def __len__(self):
        return self._con.execute("SELECT Count(*) FROM cache").fetchone()[0]
    
    
    def __getitem__(self, key):
        if not isinstance(key, six.binary_type):
            raise TypeError('Key must be bytes, but was %r' % key)
        res = self._con.execute("SELECT value FROM cache WHERE key=? "
                                "LIMIT 1", (key,)).fetchone()
        if res:
            return res[0]
        else:
            raise KeyError(key)
        
        
    def __setitem__(self, key, value):
        if not (isinstance(key, six.binary_type) and
                isinstance(value, six.binary_type)):
            raise TypeError('Keys and values must be bytes')
        with self._con:
            self._con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)",
                              (key, value))


    def __delitem__(self, key):
        if not isinstance(key, six.binary_type):
            raise TypeError('Key must be bytes, but was %r' % key)
        with self._con:
            self._con.execute("DELETE FROM cache where key=?", (key,))
    
    
    def __contains__(self, key):
        if not isinstance(key, six.binary_type):
            raise TypeError('Key must be bytes, but was %r' % key)
        return self._con.execute("SELECT EXISTS(SELECT 1 FROM cache "
                                 "WHERE key=? LIMIT 1);", (key,)).fetchone()[0]
    
    
    def __iter__(self):
        for row in self._con.execute("SELECT key FROM cache").fetchall():
            yield row[0]
            
            

class SerializedDict(collections.MutableMapping):
    """ a key value database which is stored on the disk
    This class provides hooks for converting arbitrary keys and values to
    strings, which are then stored in the database.
    """
    
    def __init__(self, key_serialization='pickle',
                 value_serialization='pickle', storage_dict=None):
        """ provides a dictionary whose keys and values are serialized
        transparently. The serialization methods are determined by
        `key_serialization` and `value_serialization`.
        
        `storage_dict` can be used to chose a different dictionary for the
            underlying storage mechanism, e.g., storage_dict = PersistentDict() 
        """
        # initialize the dictionary that actually stores the data
        if storage_dict is None:
            self._data = {}
        else:
            self._data = storage_dict
        
        # define the methods that serialize and unserialize the data
        self.serialize_key = make_serializer(key_serialization)
        self.unserialize_key = make_unserializer(key_serialization)
        self.serialize_value = make_serializer(value_serialization)
        self.unserialize_value = make_unserializer(value_serialization)
    
    
    def __len__(self):
        return len(self._data)
    
    
    def __getitem__(self, key):
        # convert key to its string representation
        key_s = self.serialize_key(key)
        # fetch the value
        value = self._data[key_s]
        # convert the value to its object representation
        return self.unserialize_value(value)
        
        
    def __setitem__(self, key, value):
        # convert key and value to their string representations
        key_s = self.serialize_key(key)
        value_s = self.serialize_value(value)
        # add the item to the dictionary
        self._data[key_s] = value_s


    def __delitem__(self, key):
        # convert key to its string representation
        key_s = self.serialize_key(key)
        # delete the item from the dictionary
        del self._data[key_s]
    
    
    def __contains__(self, key):
        # convert key to its string representation
        key_s = self.serialize_key(key)
        # check whether this items exists in the dictionary
        return key_s in self._data
    
    
    def __iter__(self):
        # iterate  dictionary
        for key_s in self._data.__iter__():
            # convert the value to its object representation
            yield self.unserialize_key(key_s)



class cached_property(object):
    """Decorator to use a function as a cached property.

    The function is only called the first time and each successive call returns
    the cached result of the first call.

        class Foo(object):

            @cached_property
            def foo(self):
                return "Cached"
                
    The data is stored in a dictionary named `_cache_properties` attached to the
    instance of each object. The cache can thus be cleared by setting
    self._cache_properties = {}

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
                cache = obj._cache_properties
            except AttributeError:
                cache = obj._cache_properties = {}
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
    


class cached_method(object):
    """ class handling the caching of results of methods """
    
    def __init__(self, factory=None, serializer='pickle', doc=None, name=None):
        """ decorator that caches method calls in a dictionary attached to the
        instances. This can be used with most classes
    
            class Foo(object):
    
                @cached_method()
                def foo(self):
                    return "Cached"
        
                @cached_method()
                def bar(self):
                    return "Cached"
                    
        
            foo = Foo()
            foo.bar()
            
            # The first call to a cached method creates the attribute
            # `foo._cache_methods`, which is a dictionary containing the
            # cache for each method.
            
        The cache can be cleared by setting foo._cache_methods = {}
        Alternatively, each cached method has a `clear_cache` method, which
        clears the cache of this particular method. In the example above we
        could thus call `foo.bar.clear_cache(foo)` to clear the cache. Note
        that the object instance has to be passed as a parameter, since the
        method `bar` is defined on the class, not the instance, i.e., we could
        also call Foo.bar.clear_cache(foo). 
            
        This class also plays together with user-supplied storage backends by 
        defining a cache factory. The cache factory should return a dict-like
        object that handles the cache for the given method.
        
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
    
        # create the function to serialize the keys
        serialize_key = make_serializer(self.serializer)
    
        @functools.wraps(method)
        def wrapper(obj, *args, **kwargs):
            # try accessing the cache
            try:
                cache = obj._cache_methods[self.name]
            except (AttributeError, KeyError) as err:
                # the cache was not initialized
                if isinstance(err, AttributeError):
                    # the cache dictionary is not even present
                    obj._cache_methods = {}
                # create cache using the right factory method
                if self.factory is None:
                    cache = {}
                else:
                    cache = getattr(obj, self.factory)(self.name)
                # store the cache in the dictionary
                obj._cache_methods[self.name] = cache
    
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
    

        def clear_cache(obj):
            """ clears the cache associated with this method """
            try:
                # try getting an initialized cache
                cache = obj._cache_methods[self.name]
                
            except (AttributeError, KeyError):
                # the cache was not initialized
                if self.factory is None:
                    # the cache would be a dictionary, but it is not yet
                    # initialized => we don't need to clear anything
                    return
                # initialize the cache, since it might open a persistent
                # database, which needs to be cleared
                cache = getattr(obj, self.factory)(self.name)
                
            # clear the cache
            cache.clear()
    
    
        # save name, e.g., to be able to delete cache later
        wrapper._cache_name = self.name
        wrapper.clear_cache = clear_cache
    
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
