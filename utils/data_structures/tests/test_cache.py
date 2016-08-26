'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

from .. import cache



class TestCache(unittest.TestCase):
    """ test collection for caching methods """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel


    def test_serializer(self):
        """ test the make_serializer and make_unserializer """
        methods = [None, 'json', 'yaml', 'pickle']
        
        # check whether yaml is actually available
        try:
            import yaml  # @UnusedImport
        except ImportError:
            methods.remove('yaml')
        
        data = [None, 1, [1, 2], {'b': 1, 'a': 2}]
        for method, data in zip(methods, data):
            encode = cache.make_serializer(method)
            decode = cache.make_unserializer(method)
            self.assertEqual(data, decode(encode(data)))
            
        self.assertRaises(ValueError,
                          lambda: cache.make_serializer('non-sense'))
        self.assertRaises(ValueError,
                          lambda: cache.make_unserializer('non-sense'))
    

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
        serializer = ['json', 'yaml', 'pickle']
        # check whether yaml is actually available
        try:
            import yaml  # @UnusedImport
        except ImportError:
            serializer.remove('yaml')
        
        for serializer in serializer:
            for cache_factory in [None, 'get_finite_dict']:
                self._test_method_cache(serializer, cache_factory)



if __name__ == "__main__":
    unittest.main()
    