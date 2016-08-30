'''
Created on Aug 29, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import os.path
import unittest
import tempfile

from .. import latex



class Test(unittest.TestCase):


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def test_numbers2latex(self):
        """ test the numbers2latex function """
        n2l = latex.numbers2latex
        self.assertEqual(n2l(1), '1')
        self.assertEqual(n2l([1.0, 1.1]), ['1', '1.1'])


    def test_number2latex(self):
        """ test the number2latex function """
        n2l = latex.number2latex
        self.assertEqual(n2l(0), '0')
        self.assertEqual(n2l(1), '1')
        self.assertEqual(n2l(1.0), '1')
        self.assertEqual(n2l(1.1), '1.1')
        
        self.assertEqual(n2l(-0), '0')
        self.assertEqual(n2l(-1), '-1')
        self.assertEqual(n2l(-10), '-10')
        self.assertEqual(n2l(-1e4), '-10^{4}')
        
        self.assertEqual(n2l(1e-3), '10^{-3}')
        self.assertEqual(n2l(1e-2), '0.01')
        self.assertEqual(n2l(1e3), '1000')
        self.assertEqual(n2l(1e4), '10^{4}')

        self.assertEqual(n2l(1.1e4), '1.1 \\times 10^{4}')

        self.assertEqual(n2l(1e-2, exponent_threshold=2), '10^{-2}')
        self.assertEqual(n2l(1e-1, exponent_threshold=2), '0.1')
        self.assertEqual(n2l(1e2, exponent_threshold=2), '100')
        self.assertEqual(n2l(1e3, exponent_threshold=2), '10^{3}')
        
        self.assertEqual(n2l(0.123456789), '0.123457')
        self.assertEqual(n2l(0.123456789, precision=2), '0.12')
        self.assertEqual(n2l(9.99e4, precision=1), '10^{5}')
        self.assertEqual(n2l(-9.99e4, precision=1), '-10^{5}')
        self.assertEqual(n2l(0.1, add_dollar=True), '$0.1$')
        
    
    def test_tex2pdf(self):
        """ test the tex2pdf function """
        tex_source = r"""\documentclass{article}
                         \begin{document}
                         Test
                         \end{document}"""
        outfile = tempfile.NamedTemporaryFile(suffix='.pdf')
        
        for use_pdflatex in (True, False):
            latex.tex2pdf(tex_source, outfile.name, use_pdflatex, verbose=False)
            self.assertGreater(os.path.getsize(outfile.name), 0)



if __name__ == "__main__":
    unittest.main()
