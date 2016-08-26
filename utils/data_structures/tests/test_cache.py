'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest
import tempfile

from .. import cache



class TestCache(unittest.TestCase):
    """ test collection for caching methods """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel


    def get_serialization_methods(self, with_none=True):
        """ returns possible methods for serialization that are supported """
        methods = ['json', 'pickle']
        
        if with_none:
            methods.append(None)
        
        # check whether yaml is actually available
        try:
            import yaml  # @UnusedImport
        except ImportError:
            pass
        else:
            methods.append('yaml')
            
        return methods


    def test_serializer(self):
        """ test the make_serializer and make_unserializer """
        methods = self.get_serialization_methods()
        data = [None, 1, [1, 2], {'b': 1, 'a': 2}]
        for method, data in zip(methods, data):
            encode = cache.make_serializer(method)
            decode = cache.make_unserializer(method)
            self.assertEqual(data, decode(encode(data)))
            
        self.assertRaises(ValueError,
                          lambda: cache.make_serializer('non-sense'))
        self.assertRaises(ValueError,
                          lambda: cache.make_unserializer('non-sense'))
    
    
    def test_DictFiniteCapacity(self):
        """ tests the DictFiniteCapacity class """
        data = cache.DictFiniteCapacity(capacity=2)
        
        data['a'] = 1
        self.assertTrue(len(data), 1)
        data['b'] = 2
        self.assertTrue(len(data), 2)
        data['c'] = 3
        self.assertTrue(len(data), 2)
        self.assertRaises(KeyError, lambda: data['a'])
        
        data.update({'d': 4})
        self.assertTrue(len(data), 2)
        self.assertRaises(KeyError, lambda: data['b'])
 
 
    def test_PersistentDict(self):
        """ tests the PersistentDict class """
        db = tempfile.NamedTemporaryFile()
        data = cache.PersistentDict(db.name)
        
        self.assertRaises(TypeError, lambda: data[1])
        self.assertRaises(TypeError, lambda: 1 in data)
        def set_int(): data['a'] = 1
        self.assertRaises(TypeError, set_int)
        def del_int(): del data[1]
        self.assertRaises(TypeError, del_int)
        
        data['a'] = '1'
        self.assertEqual(len(data), 1)
        data['b'] = '2'
        self.assertEqual(len(data), 2)
        del data['a']
        self.assertEqual(len(data), 1)
        self.assertRaises(KeyError, lambda: data['a'])
        
        data.update({'d': '4'})
        self.assertTrue(len(data), 2)
        
        # reinitialize the dictionary
        data = cache.PersistentDict(db.name)
        self.assertEqual(len(data), 2)
        self.assertEqual(data['b'], '2')
        self.assertTrue('d' in data)
        self.assertEqual({'b', 'd'}, set(data.keys()))
        self.assertEqual({'2', '4'}, set(data.values()))
        data.clear()
        self.assertEqual(len(data), 0)
        
        # reinitialize the dictionary
        data = cache.PersistentDict(db.name)
        self.assertEqual(len(data), 0)
        


    def _test_SerializedDict(self, storage, reinitialize=None,
                             key_serialization='pickle',
                             value_serialization='pickle'):
        """ tests the SerializedDict class with a particular parameter set """
        msg = 'Serializers: key: %s, value: %s' % (key_serialization,
                                                   value_serialization)
        
        data = cache.SerializedDict(key_serialization, value_serialization,
                                    storage_dict=storage)
        
        data['a'] = 1
        self.assertEqual(len(data), 1, msg=msg)
        data['b'] = 2
        self.assertEqual(data['b'], 2, msg=msg)
        self.assertEqual(len(data), 2, msg=msg)
        del data['a']
        self.assertEqual(len(data), 1, msg=msg)
        self.assertRaises(KeyError, lambda: data['a'])
        
        data.update({'d': '4'})
        self.assertEqual(len(data), 2, msg=msg)
        
        # reinitialize the storage dictionary
        if reinitialize is not None:
            data._data = reinitialize()
        self.assertEqual(len(data), 2, msg=msg)
        self.assertEqual(data['b'], 2, msg=msg)
        self.assertTrue('d' in data, msg=msg)
        self.assertEqual({'b', 'd'}, set(data.keys()), msg=msg)
        self.assertEqual({2, '4'}, set(data.values()), msg=msg)
        data.clear()
        self.assertEqual(len(data), 0, msg=msg)
        
        # reinitialize the dictionary
        if reinitialize is not None:
            data._data = reinitialize()
        self.assertEqual(len(data), 0, msg=msg)
        

    def test_SerializedDict(self):
        """ tests the SerializedDict class """
        serializers = self.get_serialization_methods(with_none=False)
        
        # test different storage types
        for storage_type in ('none', 'dict', 'persistent_dict'):
            if storage_type == 'none':
                storage = None
                reinitialize = None
                
            elif storage_type == 'dict':
                storage = {}
                reinitialize = lambda: storage
                
            elif storage_type == 'persistent_dict':
                db = tempfile.NamedTemporaryFile()
                storage = cache.PersistentDict(db.name)
                reinitialize = lambda: cache.PersistentDict(db.name)
                
            else:
                raise ValueError('Unknown storage type `%s`' % storage_type)
            
            # test different serialization methods 
            for key_serializer in serializers:
                for value_serializer in serializers:

                    self._test_SerializedDict(
                        storage=storage, reinitialize=reinitialize,
                        key_serialization=key_serializer,
                        value_serialization=value_serializer
                    )
                    if storage is not None:
                        storage.clear()
        

    def test_property_cache(self):
        """ test cached_property decorator """
        
        # create test class
        class CacheTest(object):
            """ class for testing caching """
            
            def __init__(self): self.counter = 0
            
            def get_finite_dict(self, n):
                return cache.DictFiniteCapacity(capacity=1)
            
            @property
            def uncached(self): self.counter += 1; return 1
            
            @cache.cached_property
            def cached(self): self.counter += 1; return 2    
            
        # try to objects to make sure caching is done on the instance level
        for obj in [CacheTest(), CacheTest()]:        
            # test uncached method
            self.assertEqual(obj.uncached, 1)
            self.assertEqual(obj.counter, 1)
            self.assertEqual(obj.uncached, 1)
            self.assertEqual(obj.counter, 2)
            obj.counter = 0
            
            # test cached methods
            self.assertEqual(obj.cached, 2)
            self.assertEqual(obj.counter, 1)
            self.assertEqual(obj.cached, 2)
            self.assertEqual(obj.counter, 1)
            
            
    def _test_method_cache(self, serializer, cache_factory=None):
        """ test one particular parameter set of the cached_method decorator """
        
        # create test class
        class CacheTest(object):
            """ class for testing caching """
            
            def __init__(self): self.counter = 0
            
            def get_finite_dict(self, n):
                return cache.DictFiniteCapacity(capacity=1)
            
            def uncached(self, arg): self.counter += 1; return arg
            
            @cache.cached_method(serializer=serializer, factory=cache_factory)
            def cached(self, arg): self.counter += 1; return arg    
            
            @cache.cached_method(serializer=serializer, factory=cache_factory)
            def cached_kwarg(self, a=0, b=0): self.counter += 1; return a + b
            
        # try to objects to make sure caching is done on the instance level
        for obj in [CacheTest(), CacheTest()]:        
            
            # test uncached method
            self.assertEqual(obj.uncached(1), 1)
            self.assertEqual(obj.counter, 1)
            self.assertEqual(obj.uncached(1), 1)
            self.assertEqual(obj.counter, 2)
            obj.counter = 0
            
            # test cached methods
            for method in (obj.cached, obj.cached_kwarg):
                # run twice to test clearing the cache
                for _ in (None, None):
                    # test simple caching behavior
                    self.assertEqual(method(1), 1)
                    self.assertEqual(obj.counter, 1)
                    self.assertEqual(method(1), 1)
                    self.assertEqual(obj.counter, 1)
                    self.assertEqual(method(2), 2)
                    self.assertEqual(obj.counter, 2)
                    self.assertEqual(method(2), 2)
                    self.assertEqual(obj.counter, 2)
                    
                    # test special properties of cache_factories
                    if cache_factory is None:
                        self.assertEqual(method(1), 1)
                        self.assertEqual(obj.counter, 2)
                    elif cache_factory == 'get_finite_dict':
                        self.assertEqual(method(1), 1)
                        self.assertEqual(obj.counter, 3)
                    else:
                        raise ValueError('Unknown cache_factory `%s`'
                                         % cache_factory)
        
                    obj.counter = 0
                    # clear cache to test the second run
                    method.clear_cache(obj)
            
            # test complex cached method
            self.assertEqual(obj.cached_kwarg(1, b=2), 3)
            self.assertEqual(obj.counter, 1)
            self.assertEqual(obj.cached_kwarg(1, b=2), 3)
            self.assertEqual(obj.counter, 1)
            self.assertEqual(obj.cached_kwarg(2, b=2), 4)
            self.assertEqual(obj.counter, 2)
            self.assertEqual(obj.cached_kwarg(2, b=2), 4)
            self.assertEqual(obj.counter, 2)
            self.assertEqual(obj.cached_kwarg(1, b=3), 4)
            self.assertEqual(obj.counter, 3)
            self.assertEqual(obj.cached_kwarg(1, b=3), 4)
            self.assertEqual(obj.counter, 3)
        

    def test_method_cache(self):
        """ test the cached_method decorator with several parameters """
        for serializer in self.get_serialization_methods(with_none=False):
            for cache_factory in [None, 'get_finite_dict']:
                self._test_method_cache(serializer, cache_factory)




if __name__ == "__main__":
    unittest.main()
    