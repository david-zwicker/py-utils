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
        file1.write(b'{a}\n{b}')
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


    def test_glob_alternatives_generator(self):
        """ tests the glob_alternatives_generator function """
        def alts(s):
            return list(files.pattern_alternatives(s))
        
        # test cases without groups
        self.assertSequenceEqual(alts(''), [''])
        self.assertSequenceEqual(alts('Z'), ['Z'])
        self.assertSequenceEqual(alts('abcd'), ['abcd'])
        self.assertSequenceEqual(alts('a,x'), ['a,x'])

        # test cases with a single group
        self.assertSequenceEqual(alts('{a}'), ['a'])
        self.assertSequenceEqual(alts('{a,b}'), ['a', 'b'])
        self.assertSequenceEqual(alts('{a,b,c}'), ['a', 'b', 'c'])
        self.assertSequenceEqual(alts('{a,,b}'), ['a', '', 'b'])
        self.assertSequenceEqual(alts('a{,}b'), ['ab', 'ab'])
        self.assertSequenceEqual(alts('A{a}'), ['Aa'])
        self.assertSequenceEqual(alts('A{a,b}'), ['Aa', 'Ab'])
        self.assertSequenceEqual(alts('{a}Z'), ['aZ'])
        self.assertSequenceEqual(alts('{a,b}Z'), ['aZ', 'bZ'])
        self.assertSequenceEqual(alts('A{a}Z'), ['AaZ'])
        self.assertSequenceEqual(alts('A{a,b}Z'), ['AaZ', 'AbZ'])

        # test cases with a two group
        self.assertSequenceEqual(alts('{a}{x}'), ['ax'])
        self.assertSequenceEqual(alts('{a,b}{x}'), ['ax', 'bx'])
        self.assertSequenceEqual(alts('{a}{x,y}'), ['ax', 'ay'])
        self.assertSequenceEqual(alts('{a,b}{x,y}'), ['ax', 'ay', 'bx', 'by'])
        
        # test wrongly formatted patterns
        self.assertRaises(RuntimeError, lambda: alts('{'))
        self.assertRaises(RuntimeError, lambda: alts('{{}}'))
        self.assertRaises(RuntimeError, lambda: alts('}'))


if __name__ == "__main__":
    unittest.main()
