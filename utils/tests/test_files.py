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


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


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
        
        
    def test_replace_in_file(self):
        """ tests the replace_in_file function """
        file1 = tempfile.NamedTemporaryFile()
        file2 = tempfile.NamedTemporaryFile()
        
        # write some content to file1
        file1.write('{a}\n{b}')
        file1.flush()
        self.assertEqual(open(file1.name, 'r').read(), '{a}\n{b}')
        
        files.replace_in_file(file1.name, file2.name, a=1, b=2)
        self.assertEqual(open(file2.name, 'r').read(), '1\n2')

        files.replace_in_file(file1.name, file2.name, a=5, b=5)
        self.assertEqual(open(file2.name, 'r').read(), '5\n5')

        files.replace_in_file(file1.name, a=1, b=2)
        self.assertEqual(open(file1.name, 'r').read(), '1\n2')

        files.replace_in_file(file1.name, a=5, b=5)
        self.assertEqual(open(file1.name, 'r').read(), '1\n2')

        files.replace_in_file(file1.name, file1.name, a=5, b=5)
        self.assertEqual(open(file1.name, 'r').read(), '1\n2')



if __name__ == "__main__":
    unittest.main()
