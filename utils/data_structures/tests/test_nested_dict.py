'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import os.path
import unittest
import tempfile

import h5py
import numpy as np
import six

from .. import nested_dict
from ... import misc
from ... import testing



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
        if chunk_elements is not None:
            nested_dict.LazyHDFValue.chunk_elements = chunk_elements
        if compression is not None:
            nested_dict.LazyHDFValue.compression = compression
        
        key = 'key'
        data = TestValue([1, 2, 3])
        
        value = nested_dict.LazyHDFValue(TestValue, key, self.hdf_file.name)
            
        # test simple method
        self.assertIsInstance(repr(value), six.string_types)
        yaml_str = value.get_yaml_string()
        self.assertIsInstance(yaml_str, six.string_types)
        self.assertTrue(yaml_str.startswith('@'))
        self.assertTrue(yaml_str.endswith(':' + key))
        
        # try creating class from yaml string
        with self.assertRaises(ValueError):
            nested_dict.LazyHDFValue.create_from_yaml_string('', str, '')
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
        for chunk_elements in (None, 10):
            for compression in (None, 'gzip'):
                self._test_element(chunk_elements=chunk_elements,
                                   compression=compression)
        
        
        
class TestElement(TestValue):
    """ test class describing an item in the test collection """
    
    def save_to_hdf5(self, hdf_file, key):
        """ save the data of the current burrow to an HDF5 file """
        if key in hdf_file:
            del hdf_file[key]
        hdf_file.create_dataset(key, data=self.arr)

    @classmethod
    def create_from_hdf5(cls, hdf_file, key):
        """ creates a burrow track from data in a HDF5 file """
        return cls.from_array(hdf_file[key])

        

class TestCollection(list):
    """ test class for LazyHDFCollection """
    item_class = TestElement  # the class of the single item
    hdf_attributes = {'name': 'collection'}



class TestLazyHDFCollection(unittest.TestCase):
    """ test the LazyHDFCollection class """

    
    def setUp(self):
        self.hdf_file = tempfile.NamedTemporaryFile(suffix='.hdf5')
        self.hdf_folder = os.path.dirname(self.hdf_file.name)


    def test_element(self, chunk_elements=None, compression=None):
        """ test basic functionality """
        key = 'key'
        item_cls = TestCollection.item_class
        data_list = [item_cls([1, 2, 3]), item_cls([5, 6, 7])]
        data = TestCollection(data_list)
        
        cls = nested_dict.LazyHDFCollection
        value = cls(TestCollection, key, self.hdf_file.name)

        # test simple method
        self.assertIsInstance(repr(value), six.string_types)
        yaml_str = value.get_yaml_string()
        self.assertIsInstance(yaml_str, six.string_types)
        self.assertTrue(yaml_str.startswith('@'))
        self.assertTrue(yaml_str.endswith(':' + key))
        
        # try creating class from yaml string
        with self.assertRaises(ValueError):
            cls.create_from_yaml_string('', str, '')
        value2 = cls.create_from_yaml_string(yaml_str, TestCollection,
                                             self.hdf_folder)
        
        self.assertEqual(value, value2)
        value.set_hdf_folder(self.hdf_folder)
        self.assertEqual(value, value2)
        
        # try creating from data and storing to hdf5
        value3 = cls.create_from_data(key, data, self.hdf_file.name)
        self.assertEqual(value, value3)
        value4 = cls.create_from_data(key, data, self.hdf_file.name)
        self.assertEqual(value3, value4)
        
        data2 = value.load()
        self.assertEqual(data, data2)
        data3 = value2.load()
        self.assertEqual(data, data3)
        
        

class TestNestedDict(unittest.TestCase):
    """ test the NestedDict class """

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel
    dict_cls = nested_dict.NestedDict


    def test_basics(self):
        """ tests miscellaneous functions """
        d = self.dict_cls({'a': {'b': {}}, 'c': 1})
        
        self.assertIsInstance(repr(d), str)
        
        stream = six.StringIO()
        with misc.RedirectedStdout(stream):
            d.pprint()
        self.assertGreater(len(stream.getvalue()), 0)


    def test_getting_data(self):
        """ tests that are about retrieving data """
        d = self.dict_cls({'a': {'b': {}}, 'c': 1})    
        self.assertEqual(d['a'], self.dict_cls({'b': {}}))
        self.assertEqual(d['a/b'], {})
        self.assertEqual(d['c'], 1)
        
        with self.assertRaises(KeyError):
            d['z']
        with self.assertRaises(KeyError):
            d['a/z']
        with self.assertRaises(KeyError):
            d['c/z']


    def test_membership(self):
        """ tests that test membership """
        d = self.dict_cls({'a': {'b': {}}, 'c': 1})    

        self.assertIn('a', d)
        self.assertIn('a/b', d)
        self.assertIn('c', d)

        self.assertNotIn('z', d)
        self.assertNotIn('a/z', d)
        self.assertNotIn('c/z', d)
        
        
    def test_setting_data(self):
        """ tests that are about setting data """
        d = self.dict_cls({'a': {'b': {}}})
        
        d['a/c'] = 2
        self.assertEqual(d['a/c'], 2)
        
        d['e/f'] = 3
        self.assertEqual(d['e/f'], 3)
        
        self.assertEqual(d.to_dict(), {'a': {'b': {}, 'c': 2}, 'e': {'f': 3}})
        self.assertEqual(d.to_dict(flatten=True), {'a/c': 2, 'e/f': 3})
        
        d = self.dict_cls({'a': {'b': {}}})    
        d['a'] = 2
        self.assertEqual(d, self.dict_cls({'a': 2}))

        with self.assertRaises(TypeError):
            d['a/b'] = 2
            
        r = d.create_child('f', {'1': 2})
        self.assertEqual(r, self.dict_cls({'1': 2}))
        self.assertEqual(d, self.dict_cls({'a': 2, 'f': {'1': 2}}))
        
        d = self.dict_cls({'a': {'b': 1}})
        d.from_dict({'a/c': 2})
        self.assertEqual(d, self.dict_cls({'a': {'b': 1, 'c': 2}}))
        d.from_dict({'a': {'c': 3}})
        self.assertEqual(d, self.dict_cls({'a': {'b': 1, 'c': 3}}))
            
            
    def test_deleting_data(self):
        """ tests that are about deleting data """
        d = self.dict_cls({'a': {'b': 1, 'c': 2}, 'd': 3})

        with self.assertRaises(KeyError):
            del d['g']
        
        with self.assertRaises(KeyError):
            del d['d/z']
        
        del d['d']
        self.assertEqual(d, self.dict_cls({'a': {'b': 1, 'c': 2}}))
        
        del d['a/c']
        self.assertEqual(d, self.dict_cls({'a': {'b': 1}}))
        
        del d['a']
        self.assertEqual(d, self.dict_cls())
        

    def test_iterators(self):
        """ test iterating over the data """
        d = self.dict_cls({'a': {'b': 1}, 'c': 2})
        
        six.assertCountEqual(self, d.iterkeys(), ['a', 'c'])    
        six.assertCountEqual(self, d.iterkeys(flatten=True), ['a/b', 'c'])    
        
        six.assertCountEqual(self, d.itervalues(), [self.dict_cls({'b': 1}), 2])    
        six.assertCountEqual(self, d.itervalues(flatten=True), [1, 2])    
        
        six.assertCountEqual(self, d.iteritems(),
                             [('a', self.dict_cls({'b': 1})), ('c', 2)])    
        six.assertCountEqual(self, d.iteritems(flatten=True),
                             [('a/b', 1), ('c', 2)]) 
        
        # test some exceptions
        with self.assertRaises(TypeError):
            list(self.dict_cls({1: {2: 3}}).iterkeys(flatten=True))   
        with self.assertRaises(TypeError):
            list(self.dict_cls({1: {2: 3}}).iteritems(flatten=True))   
        
        
    def test_conversion(self):
        """ test the conversion of dictionaries """
        d = self.dict_cls({'a': {'b': 1}, 'c': 2})

        d2 = d.copy()
        self.assertEqual(d2, d)
        d2['c'] = 3
        self.assertNotEqual(d2, d)

        d3 = self.dict_cls(d.to_dict())
        self.assertEqual(d3, d)
        d3 = self.dict_cls(d.to_dict(flatten=True))
        self.assertEqual(d3, d)
        
        d = self.dict_cls({1: {2: 3}})
        with self.assertRaises(TypeError):
            d.to_dict(flatten=True)
        



class TestLazyNestedDict(TestNestedDict):
    """ test the LazyNestedDict class """

    _multiprocess_can_split_ = True  # let nose know that tests can run parallel
    dict_cls = nested_dict.LazyNestedDict
    

    def setUp(self):
        self.hdf_file = tempfile.NamedTemporaryFile(suffix='.hdf5')
        self.hdf_folder = os.path.dirname(self.hdf_file.name)


    def test_simple(self):
        """ test the functionality """
        key = 'key'
        data = TestValue([1, 2, 3])
        
        # try simple storage
        value = nested_dict.LazyHDFValue.create_from_data(key, data,
                                                          self.hdf_file.name)
        
        # modify original data
        data.arr = np.arange(5)
        data2 = TestValue([1, 2, 3])
        
        d = nested_dict.LazyNestedDict({'a': value})
        
        self.assertEqual(d.get_item('a', load_data=False), value)
        self.assertEqual(d.get_item('a', load_data=True), data2)
        self.assertEqual(d.get_item('a', load_data=False), data2)

        # try again, but with different data 
        value = nested_dict.LazyHDFValue.create_from_data(key, data,
                                                          self.hdf_file.name)
        
        d = nested_dict.LazyNestedDict({'a': value})
        
        self.assertEqual(d.get_item('a', load_data=False), value)
        self.assertEqual(d.get_item('a', load_data=True), data)
        self.assertEqual(d.get_item('a', load_data=False), data)

        # try again, but corrupt the database in between  
        value = nested_dict.LazyHDFValue.create_from_data(key, data,
                                                          self.hdf_file.name)
        
        with h5py.File(self.hdf_file.name, 'r+') as hdf_db:
            del hdf_db[key]
        
        d = nested_dict.LazyNestedDict({'a': value})
        
        self.assertEqual(d.get_item('a', load_data=False), value)
        with self.assertRaises(nested_dict.LazyLoadError):
            d.get_item('a', load_data=True)



class TestGeneral(unittest.TestCase, testing.WarnAssertionsMixin):
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
                    
    
    def test_prepare_data_for_yaml(self):
        """ test several types """
        valid_data = [np.arange(5),
                      np.float64(1), np.float128(1),
                      np.uint8(3), np.int64(3),
                      {'1': 2},
                      [1, 2, 3], (1, 2, 3), {1, 2, 3},
                      None, 1, 'str', 1.2, True, False]
        
        for data in valid_data:
            nested_dict.prepare_data_for_yaml(data)
                                
        invalid_data = [TestValue([1, 2])]
        for data in invalid_data:
            with self.assertWarnings(['unknown instance']):
                nested_dict.prepare_data_for_yaml(data)



if __name__ == "__main__":
    unittest.main()
