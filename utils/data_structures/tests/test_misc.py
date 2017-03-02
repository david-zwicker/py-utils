'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import csv
import os
import unittest
import tempfile

from .. import misc



class TestMisc(unittest.TestCase):


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def test_transpose_list_of_dicts(self):
        """ test transpose_list_of_dicts function """
        data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 6}]
        res = misc.transpose_list_of_dicts(data)
        self.assertEqual(res, {'a': [1, 5], 'b': [2, 6]})

        data = [{'a': 1, 'b': 2}, {'a': 5}]
        res = misc.transpose_list_of_dicts(data)
        self.assertEqual(res, {'a': [1, 5], 'b': [2, None]})

        data = [{'a': 1, 'b': 2}, {'a': 5}]
        res = misc.transpose_list_of_dicts(data, missing=-1)
        self.assertEqual(res, {'a': [1, 5], 'b': [2, -1]})


    def test_save_dict_to_csv(self):
        """ test the save_to_dict function """
        tmpfile = tempfile.NamedTemporaryFile(mode='w+')
        
        data = {'a': ['1', '2'], 'b': ['4', '5']}
        misc.save_dict_to_csv(data, tmpfile.name)
        
        reader = csv.DictReader(tmpfile)
        read = misc.transpose_list_of_dicts([row for row in reader])
        
        data_expected = data.copy()
        data_expected[''] = ['0', '1']
        self.assertEqual(data_expected, read)


    def test_save_dict_to_csv_units(self):
        """ test the save_to_dict function with units """
        try:
            import pint
        except ImportError:
            # Units are not available and we thus don't test this
            return

        ureg = pint.UnitRegistry()
        tmpfile = tempfile.NamedTemporaryFile(mode='w+')
        
        data = {'a': [1 * ureg.meter, 2 * ureg.meter],
                'b': [4, 5] * ureg.second}
        misc.save_dict_to_csv(data, tmpfile.name)
        
        reader = csv.DictReader(tmpfile)
        read = misc.transpose_list_of_dicts([row for row in reader])
        
        data_expected = {'': ['0', '1'], 'a [meter]': ['1', '2'],
                         'b [second]': ['4', '5']}
        self.assertEqual(data_expected, read)
        
        data = {'a': [1 * ureg.meter, 1]}
        self.assertRaises(AttributeError,
                          lambda: misc.save_dict_to_csv(data, ''))
        
        
    def test_OmniContainer(self):
        """ test the OmniContainer class """
        container = misc.OmniContainer()
        del container['item']
        self.assertTrue(container)
        self.assertTrue('anything' in container)
        self.assertTrue(isinstance(repr(container), str))
        
        
    def test_yaml_database(self):
        """ test the yaml_database function """
        import portalocker
        db_file = tempfile.NamedTemporaryFile(delete=False).name
        
        testcases = {
            'legacy':  lambda: misc.yaml_database(db_file),
            'simple':  lambda: misc.SimpleDatabase(db_file, locking=False),
            'locking': lambda: misc.SimpleDatabase(db_file, locking=True),
        }
        
        for msg, database in testcases.items():
            with database() as db:
                self.assertEqual(db, {}, msg=msg)
                db['a'] = 1
                
            with database() as db:
                self.assertEqual(db, {'a': 1}, msg=msg)
                db['b'] = 2
                
            with database() as db:
                self.assertEqual(db, {'a': 1, 'b': 2}, msg=msg)
    
            os.remove(db_file) # simulate removing the file
            with database() as db:
                self.assertEqual(db, {}, msg=msg)        
            
        # test that database cannot be opened a second time if locked
        with misc.SimpleDatabase(db_file, locking=True):
            def open_db():
                with misc.SimpleDatabase(db_file, locking=True):
                    pass
            self.assertRaises(portalocker.LockException, open_db)
            
        # clean-up
        os.remove(db_file)


if __name__ == "__main__":
    unittest.main()
