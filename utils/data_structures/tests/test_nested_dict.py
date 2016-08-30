'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import os.path
import unittest
import tempfile

import numpy as np
import six

from .. import nested_dict



class TestGeneral(unittest.TestCase):
    """ test general functions of the module """


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel
        
        
    def test_get_chunk_size(self):
        """ test the get_chunk_size function """
        for _ in range(10):
            shape = np.random.randint(1, 10, size=10)
            size = np.prod(shape)
            for _ in range(10):
                num_elements = np.random.randint(1, size)
                chunks = nested_dict.get_chunk_size(shape, num_elements)
                self.assertLessEqual(np.prod(chunks), num_elements)
                for c, s in zip(chunks, shape):
                    self.assertLessEqual(c, s)



class TestValue(object):
    """ test class for LazyHDFValue """
    
    hdf_attributes = {'name': 'value'}
    
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.double)
        
    def __eq__(self, other):
        return np.array_equal(self.arr, other.arr)        
        
    @classmethod
    def from_array(cls, arr):
        return cls(arr)
    
    def to_array(self):
        return self.arr



class TestLazyHDFValue(unittest.TestCase):
    """ test the LazyHDFValue class """

    
    def setUp(self):
        self.hdf_file = tempfile.NamedTemporaryFile(suffix='.hdf5')
        self.hdf_folder = os.path.dirname(self.hdf_file.name)


    def _test_element(self, chunk_elements=None, compression=None):
        """ test basic functionality """
        key = 'key'
        data = TestValue([1, 2, 3])
        
        value = nested_dict.LazyHDFValue(TestValue, key, self.hdf_file.name)
        if chunk_elements is not None:
            value.chunk_elements = chunk_elements
        if compression is not None:
            value.compression = compression
            
        # test simple method
        self.assertIsInstance(repr(value), six.string_types)
        yaml_str = value.get_yaml_string()
        self.assertIsInstance(yaml_str, six.string_types)
        self.assertTrue(yaml_str.startswith('@'))
        self.assertTrue(yaml_str.endswith(':' + key))
        
        # try creating class from yaml string
        def create_wrong():
            return nested_dict.LazyHDFValue.create_from_yaml_string('', str, '')
        self.assertRaises(ValueError, create_wrong)
        value2 = nested_dict.LazyHDFValue.create_from_yaml_string(
                                          yaml_str, TestValue, self.hdf_folder)
        
        self.assertEqual(value, value2)
        value.set_hdf_folder(self.hdf_folder)
        self.assertEqual(value, value2)
        
        # try creating from data and storing to hdf5
        value3 = nested_dict.LazyHDFValue.create_from_data(key, data,
                                                           self.hdf_file.name)
        self.assertEqual(value, value3)
        value4 = nested_dict.LazyHDFValue.create_from_data(key, data,
                                                           self.hdf_file.name)
        self.assertEqual(value3, value4)
        
        data2 = value.load()
        self.assertEqual(data, data2)
        data3 = value2.load()
        self.assertEqual(data, data3)
            
            
    def test_element(self):
        self._test_element()
        



class TestNestedDict(unittest.TestCase):
    """ test the NestedDict class """

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def test_simple(self):
        """ simple tests that reflect normal usage """
        d = nested_dict.NestedDict({'a': {'b': {}}})    
        self.assertEqual(d['a/b'], {})
        self.assertRaises(KeyError, lambda: d['c'])
        self.assertRaises(KeyError, lambda: d['a/c'])
        
        d['a/c'] = 2
        self.assertEqual(d['a/c'], 2)
        
        d['e/f'] = 3
        self.assertEqual(d['e/f'], 3)
        
        self.assertEqual(d.to_dict(), {'a': {'b': {}, 'c': 2}, 'e': {'f': 3}})
        self.assertEqual(d.to_dict(flatten=True), {'a/c': 2, 'e/f': 3})



if __name__ == "__main__":
    unittest.main()
    