'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import csv
import unittest
import tempfile

import pint

from ..misc import *  # @UnusedWildImport




class Test(unittest.TestCase):


    _multiprocess_can_split_ = True #< let nose know that tests can run parallel


    def test_transpose_list_of_dicts(self):
        """ test transpose_list_of_dicts function """
        data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 6}]
        res = transpose_list_of_dicts(data)
        self.assertEquals(res, {'a': [1, 5], 'b': [2, 6]})

        data = [{'a': 1, 'b': 2}, {'a': 5}]
        res = transpose_list_of_dicts(data)
        self.assertEquals(res, {'a': [1, 5], 'b': [2, None]})

        data = [{'a': 1, 'b': 2}, {'a': 5}]
        res = transpose_list_of_dicts(data, missing=-1)
        self.assertEquals(res, {'a': [1, 5], 'b': [2, -1]})


    def test_save_dict_to_csv(self):
        """ test the save_to_dict function """
        tmpfile = tempfile.NamedTemporaryFile()
        
        data = {'a': ['1', '2'], 'b': ['4', '5']}
        save_dict_to_csv(data, tmpfile.name)
        
        reader = csv.DictReader(tmpfile)
        read = transpose_list_of_dicts([row for row in reader])
        
        data_expected = data.copy()
        data_expected[''] = ['0', '1']
        self.assertEquals(data_expected, read)


    def test_save_dict_to_csv_units(self):
        """ test the save_to_dict function with units """
        ureg = pint.UnitRegistry()
        tmpfile = tempfile.NamedTemporaryFile()
        
        data = {'a': [1 * ureg.meter, 2 * ureg.meter],
                'b': [4, 5] * ureg.second}
        save_dict_to_csv(data, tmpfile.name)
        
        reader = csv.DictReader(tmpfile)
        read = transpose_list_of_dicts([row for row in reader])
        
        data_expected = {'': ['0', '1'], 'a [meter]': ['1', '2'],
                         'b [second]': ['4', '5']}
        self.assertEquals(data_expected, read)
        
        data = {'a': [1 * ureg.meter, 1]}
        self.assertRaises(AttributeError, lambda: save_dict_to_csv(data, ''))
        
        
    def test_OmniContainer(self):
        """ test the OmniContainer class """
        container = OmniContainer()
        del container['item']
        self.assertTrue(container)
        self.assertTrue('anything' in container)
        


if __name__ == "__main__":
    unittest.main()