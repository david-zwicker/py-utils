'''
Created on Aug 29, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging
import unittest
import warnings

from .. import misc



class TestMisc(unittest.TestCase):


    _multiprocess_can_split_ = True #< let nose know that tests can run parallel


    def test_DummyFile(self):
        """ test the DummyFile class """
        misc.DummyFile().write("a")
        

    def test_copy_func(self):
        """ test the copy_func function """
        def func(a, b):
            """ test function """
            return a + b
        
        func_cp = misc.copy_func(func)
        self.assertEqual(func_cp(1, 2), 3)
        self.assertEqual(func_cp.__doc__, func.__doc__)
        self.assertEqual(func_cp.__name__, func.__name__)


    def test_DeprecationHelper(self):
        """ test the DeprecationHelper class """
        # create a class that uses the deprecation helper
        @misc.DeprecationHelper
        class Class(object):
            def __init__(self): self.value = 1
            @classmethod
            def create(cls): return Class()
            def get_value(self): return self.value
        
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            obj = Class.create()
            # test whether the class still works
            self.assertEqual(obj.get_value(), 1)
            # test whether two warning were emitted
            self.assertEqual(len(messages), 2)
            for w in messages:
                self.assertTrue(issubclass(w.category, DeprecationWarning))
                self.assertIn("class was renamed", str(w.message))
            
            
    def test_display_progress(self):
        """ test display_progress function """
        misc.display_progress([])


    def test_get_loglevel_from_name(self):
        """ test the get_loglevel_from_name function """
        # the function should just pass down integers
        self.assertEqual(misc.get_loglevel_from_name(1), 1)
        
        # strings should be translated to their respective log-level
        self.assertEqual(misc.get_loglevel_from_name('info'), logging.INFO)
        self.assertEqual(misc.get_loglevel_from_name('warn'), logging.WARN)
        self.assertEqual(misc.get_loglevel_from_name('debug'), logging.DEBUG)
        
        self.assertRaises(ValueError, lambda: misc.get_loglevel_from_name('a'))


    def test_deprecated(self):
        """ test the deprecated decorator """
        @misc.deprecated
        def function():
            return 1, None
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # test whether the function still works
            self.assertEqual(function(), (1, None))
            # test whether a warning was emitted
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn("deprecated", str(w[-1].message))
            

    def test_unique_based_on_id(self):
        """ test the unique_based_on_id function """
        unique = misc.unique_based_on_id
        self.assertSequenceEqual(unique([]), [])
        self.assertSequenceEqual(unique([1, 2, 3]), [1, 2, 3])
        self.assertSequenceEqual(unique([1, 1, 3]), [1, 3])
        self.assertSequenceEqual(unique([[], [], 3]),[[], [], 3])



if __name__ == "__main__":
    unittest.main()