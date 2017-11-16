'''
Created on Sep 11, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains functions that can be used to manage cache structures
'''

from __future__ import division

import collections
import functools
import logging
import os
import sys

import six

import numpy as np



def make_hash_key(obj):
    """ return hash also for mutable objects """
    try:
        return hash(obj)
    except TypeError:
        if isinstance(obj, list):
            return hash(make_hash_key(v) for v in obj)
        elif isinstance(obj, set):
            return hash(frozenset(make_hash_key(v) for v in obj))
        elif isinstance(obj, dict):
            return hash(frozenset((k, make_hash_key(v))
                                  for k, v in six.iteritems(obj)))
        raise
    
    

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
        return lambda s: pickle.dumps(s, protocol=pickle.HIGHEST_PROTOCOL)

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



class _class_cache(object):
    """ class handling the caching of results of methods and properties """
    
    def __init__(self, factory=None, extra_args=None, ignore_args=None,
                 hash_function='pickle', doc=None, name=None):
        """ decorator that caches calls in a dictionary attached to the
        instances. This can be used with most classes
    
            class Foo(object):
    
                @cached_property()
                def property(self):
                    return "Cached property"
        
                @cached_method()
                def method(self):
                    return "Cached method"
                    
        
            foo = Foo()
            foo.property
            foo.method()
            
            # The first call to a cached method creates the attribute
            # `foo._cache_methods`, which is a dictionary containing the
            # cache for each method.
            
        The cache can be cleared by setting foo._cache_methods = {} if the cache
        factor is a simple dict, i.e, if `factory` == None.        
        Alternatively, each cached method has a `clear_cache_of_obj` method,
        which clears the cache of this particular method. In the example above
        we could thus call `foo.bar.clear_cache_of_obj(foo)` to clear the cache.
        Note that the object instance has to be passed as a parameter, since the
        method `bar` is defined on the class, not the instance, i.e., we could
        also call Foo.bar.clear_cache_of_obj(foo).
        
        For convenience there is also the class decorator
        `add_clear_cache_method` that adds a method `clear_cache` that can be
        used to clear the caches of all methods of the class and its subclasses
        
        Additionally, `extra_args` can specify a list of properties that are 
        added to the cache key. They are then treated as if they are supplied as
        arguments to the method. This is important to include when the result of
        a method depends not only on method arguments but also on instance
        properties. Conversely, the keyword arguments listed in `ignore_args`
        are ignored in the cache key.
            
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
        self.extra_args = extra_args
        self.hash_function = hash_function
        self.name = name

        # setup the ignored arguments
        if ignore_args is not None:
            if isinstance(ignore_args, six.string_types):
                ignore_args = [ignore_args]
            self.ignore_args = set(ignore_args)
        else:
            self.ignore_args = None

        # check whether the decorator has been applied correctly
        if callable(factory):
            class_name = self.__class__.__name__
            raise ValueError('Missing function call. Call this decorator as '
                             '@{0}() instead of @{0}'.format(class_name))
            
        else:
            self.factory = factory
        
    
    def _get_clear_cache_method(self):
        """ return a method that can be attached to classes to clear the cache
        of the wrapped method """
        
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
            
        return clear_cache
    
        
    def _get_wrapped_function(self, func):
        """ return the wrapped method, which implements the cache """
        
        if self.name is None:
            self.name = func.__name__
    
        # create the function to serialize the keys
        hash_key = make_serializer(self.hash_function)
    
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            # try accessing the cache
            try:
                cache = obj._cache_methods[self.name]
            except (AttributeError, KeyError) as err:
                # the cache was not initialized
                wrapper._logger.debug('Initialize the cache `%s`', self.name)
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
            if self.ignore_args:
                kwargs_key = {k: v for k, v in six.iteritems(kwargs)
                              if k not in self.ignore_args}
                func_args = [args, kwargs_key]
            else:
                func_args = [args, kwargs]
                
            if self.extra_args:
                for extra_arg in self.extra_args:
                    func_args.append(getattr(obj, extra_arg))
                    
            cache_key = hash_key(tuple(func_args))
    
            try:
                # try loading the results from the cache
                result = cache[cache_key]
            except KeyError:
                # if this failed, compute and store the results
                wrapper._logger.debug('Cache missed. Compute result for method '
                                      '`%s`', self.name)
                result = func(obj, *args, **kwargs)
                cache[cache_key] = result
            return result
        
        # initialize the logger
        wrapper._logger = logging.getLogger(__name__)
        
        return wrapper



class cached_property(_class_cache):
    """Decorator to use a function as a cached property.

    The function is only called the first time and each successive call returns
    the cached result of the first call.

        class Foo(object):

            @cached_property
            def foo(self):
                return "Cached"
                
    The data is stored in a dictionary named `_cache_methods` attached to the
    instance of each object. The cache can thus be cleared by setting
    self._cache_methods = {}

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.
    """

    def __call__(self, method):
        """ apply the cache decorator to the property """
        # save name, e.g., to be able to delete cache later
        self._cache_name = self.name
        self.clear_cache_of_obj = self._get_clear_cache_method()
        self.func = self._get_wrapped_function(method)
    
        self.__doc__ = self.func.__doc__
        self.__name__ = self.func.__name__
        self.__module__ = self.func.__module__
        return self
    
    
    def __get__(self, obj, owner):
        """ call the method to obtain the result for this property """
        if obj is None:
            return self

        return self.func(obj)
    


class cached_method(_class_cache):
    """ class handling the caching of results of methods """
    
    def __call__(self, method):
        """ apply the cache decorator to the method """
        
        wrapper = self._get_wrapped_function(method)
    
        # save name, e.g., to be able to delete cache later
        wrapper._cache_name = self.name
        wrapper.clear_cache_of_obj = self._get_clear_cache_method()
    
        return wrapper



def add_clear_cache_method(cls):
    """ a class decorator that adds a clear_cache method to the class """
    # gather the methods that need to be cleared
    methods_with_cache = []
    for method_name in dir(cls):
        if method_name.startswith('__'):
            continue
        
        method = getattr(cls, method_name)
        if hasattr(method, 'clear_cache_of_obj'):
            methods_with_cache.append(method)

    # add the actual method for clearing the cache            
    def clear_cache(self):
        """ clears the cache of all methods """
        for method in methods_with_cache:
            method.clear_cache_of_obj(self)
    cls.clear_cache = clear_cache
            
    return cls



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
