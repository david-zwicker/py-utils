'''
Created on Aug 29, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest
import tempfile

from .. import mathematica



class TestMathematica(unittest.TestCase):


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def setUp(self):
        """ find mathematica """
        try:
            self.m = mathematica.Mathematica()
        except RuntimeError:
            raise unittest.SkipTest('Mathematica cannot be found.')


    def test_run_code(self):
        """ test running code with Mathematica """
        res, err = self.m.run_code(b'1 + 2')
        self.assertEqual(err, b'')  # no message on stderr
        cells = self.m.extract_output_cells(res)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[1], b'3')
        self.assertEqual(self.m.extract_output_cell(res, 1), b'3')
        
        code = b'Integrate[1, {x, y, z} \[Element] Sphere[]]'
        res, err = self.m.run_code(code)
        self.assertEqual(err, b'')  # no message on stderr
        cells = self.m.extract_output_cells(res)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[1], b'4 Pi')
        self.assertEqual(self.m.extract_output_cell(res, 1), b'4 Pi')
        
        
    def test_parse_output(self):
        """ test parsing the output of a more complicated expression """
        res, err = self.m.run_code(b'1\n2\n3\n')
        self.assertEqual(err, b'')  # no message on stderr
        cells = self.m.extract_output_cells(res)
        self.assertEqual(len(cells), 3)
        for value in (1, 2, 3):
            self.assertEqual(cells[value], str(value).encode('utf-8'))
        

    def test_run_file(self):
        """ test running code with Mathematica """
        script = tempfile.NamedTemporaryFile()
        script.write(b'Print[1 + 2]\nExit[]')
        script.flush()
        
        res, err = self.m.run_script(script.name)
        self.assertEqual(res, b'3\n')
        self.assertEqual(err, b'')  # no message on stderr



if __name__ == "__main__":
    unittest.main()
