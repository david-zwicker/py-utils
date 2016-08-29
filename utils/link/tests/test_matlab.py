'''
Created on Aug 29, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest
import tempfile

from .. import matlab



class TestMatlab(unittest.TestCase):


    _multiprocess_can_split_ = True #< let nose know that tests can run parallel


    def test_run_code(self):
        """ test running code with Matlab """
        m = matlab.Matlab()
        
        res, err = m.run_code('disp(1 + 2)')
        self.assertEqual(err, '') #< no message on stderr
        cells = m.extract_output_cells(res)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].rstrip(), '3')
        

    def test_run_file(self):
        """ test running code with Matlab """
        script = tempfile.NamedTemporaryFile(suffix='.m')
        script.write('disp(1 + 2)')
        script.flush()
         
        m = matlab.Matlab()
         
        res, err = m.run_script(script.name)
        self.assertEqual(err, '') #< no message on stderr
        cells = m.extract_output_cells(res)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].rstrip(), '3')



if __name__ == "__main__":
    unittest.main()