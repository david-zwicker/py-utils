'''
Created on Aug 29, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest
import tempfile

from .. import matlab



class TestMatlab(unittest.TestCase):


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def setUp(self):
        """ find matlab """
        try:
            self.m = matlab.Matlab()
        except RuntimeError:
            raise unittest.SkipTest('Matlab cannot be found.')


    def test_run_code(self):
        """ test running code with Matlab """
        res, err = self.m.run_code(b'disp(1 + 2)')
        self.assertEqual(err, b'')  # no message on stderr
        cells = self.m.extract_output_cells(res)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].rstrip(), b'3')
        

    def test_run_file(self):
        """ test running code with Matlab """
        script = tempfile.NamedTemporaryFile(suffix='.m')
        script.write(b'disp(1 + 2)')
        script.flush()
         
        res, err = self.m.run_script(script.name)
        self.assertEqual(err, b'')  # no message on stderr
        cells = self.m.extract_output_cells(res)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].rstrip(), b'3')



if __name__ == "__main__":
    unittest.main()
