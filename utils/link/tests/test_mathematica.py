'''
Created on Aug 29, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest
import tempfile

from .. import mathematica



class TestMathematica(unittest.TestCase):


    _multiprocess_can_split_ = True #< let nose know that tests can run parallel


    def test_run_code(self):
        """ test running code with Mathematica """
        m = mathematica.Mathematica()
        
        res, err = m.run_code('1 + 2')
        self.assertEqual(err, '') #< no message on stderr
        cells = m.extract_output_cells(res)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[1], '3')
        self.assertEqual(m.extract_output_cell(res, 1), '3')
        
        res, err = m.run_code('Integrate[1, {x, y, z} \[Element] Sphere[]]')
        self.assertEqual(err, '') #< no message on stderr
        cells = m.extract_output_cells(res)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[1], '4 Pi')
        self.assertEqual(m.extract_output_cell(res, 1), '4 Pi')
        
        
    def test_parse_output(self):
        """ test parsing the output of a more complicated expression """
        m = mathematica.Mathematica()
        
        res, err = m.run_code('1\n2\n3\n')
        self.assertEqual(err, '') #< no message on stderr
        cells = m.extract_output_cells(res)
        self.assertEqual(len(cells), 3)
        for value in (1, 2, 3):
            self.assertEqual(cells[value], str(value))
        

    def test_run_file(self):
        """ test running code with Mathematica """
        script = tempfile.NamedTemporaryFile()
        script.write('Print[1 + 2]\nExit[]')
        script.flush()
        
        m = mathematica.Mathematica()
        
        res, err = m.run_script(script.name)
        self.assertEqual(res, '3\n')
        self.assertEqual(err, '') #< no message on stderr



if __name__ == "__main__":
    unittest.main()