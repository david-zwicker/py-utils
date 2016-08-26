'''
Created on Aug 26, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import os
import unittest
import tempfile

from .. import files



class TestFiles(unittest.TestCase):


    _multiprocess_can_split_ = True #< let nose know that tests can run parallel


    def test_change_directory(self):
        """ tests the context manager change_directory """
        folder = os.path.realpath(tempfile.gettempdir())
        cwd = os.getcwd()
        with files.change_directory(folder):
            self.assertEqual(folder, os.getcwd())
        self.assertEqual(cwd, os.getcwd())


    def test_ensure_directory_exists(self):
        """ tests the ensure_directory_exists function """
        # create temporary name
        path = tempfile.mktemp()
        self.assertFalse(os.path.exists(path))
        # create the folder
        files.ensure_directory_exists(path)
        self.assertTrue(os.path.exists(path))
        # check that a second call has the same result
        files.ensure_directory_exists(path)
        self.assertTrue(os.path.exists(path))
        # remove the folder again
        os.rmdir(path)
        self.assertFalse(os.path.exists(path))



if __name__ == "__main__":
    unittest.main()