"""
Python module defining classes for creating more versatile figures

This code and information is provided 'as is' without warranty of any kind,
either express or implied, including, but not limited to, the implied
warranties of non-infringement, merchantability or fitness for a particular
purpose.
"""

from __future__ import division

import logging
import os.path
import shutil
import subprocess as sp
import tempfile

import numpy as np

from .. import files



def numbers2latex(values, **kwargs):
    """
    Converts a list of numbers into a representation nicely displayed by latex.
    Additional parameters are passed on to `number2latex`
    """

    # apply function to all parts of a list recursively
    if hasattr(values, '__iter__'):
        return [numbers2latex(v, **kwargs) for v in values]
    else:
        return number2latex(values, **kwargs)



def number2latex(val, **kwargs):
    """ Converts a number into a representation nicely displayed by latex """

    # apply function to all parts of a potential list
    if hasattr(val, '__iter__'):
        print('Using the iterative version of `number2latex` is deprecated')
        return [number2latex(v, **kwargs) for v in val]

    # read parameters
    exponent_threshold = kwargs.pop('exponent_threshold', 3)
    add_dollar = kwargs.pop('add_dollar', False)
    precision = kwargs.pop('precision', None)

    # represent the input as mantissa + exponent
    val = float(val)
    if val == 0:
        exponent = 0
    else:
        exponent = int(np.log10(abs(val)))
        if exponent < 0:
            exponent -= 1
    mantissa = val/10**exponent

    # process these values further
    if precision is not None:
        mantissa = round(mantissa, precision)

    # distinguish different format cases
    if mantissa == 0:
        res = '0'

    elif abs(exponent) > exponent_threshold:

        # write mantissa
        res = "%g" % mantissa

        # handle special mantissa that can be omitted
        if res == '1':
            res = ''
        elif res == '-1':
            res = '-'
        elif res == '10':
            res = ''
            exponent += 1
        elif res == '-10':
            res = '-'
            exponent += 1
        else:
            res += r' \times '

        res += '10^{%d}' % (exponent)

    elif precision is not None:
        res = '%g' % round(val, precision - exponent)

    else:
        res = '%g' % val

    # check, whether to enclose the expression in dollar signs
    if add_dollar:
        res = '$%s$' % res

    return res



def tex2pdf(tex_source, outfile, use_pdflatex=True):
    """ takes latex source code and compiles it into a PDF file
    The resulting file will be copied to `outfile`. The functions returns the
    accumulated output of all function calls.
    
    `use_pdflatex` determines whether pdflatex is used to compile the source
    """
    # create temporary working directory
    tmp = tempfile.mkdtemp()
    output = ('Compiling TeX document in folder `%s`\n' % tmp).encode('utf-8')

    def call(cmd):
        return sp.check_output(cmd, stderr=sp.STDOUT)

    # create PDF
    with files.change_directory(tmp):
        f = open('document.tex', "w")
        f.write(tex_source)
        f.close()

        if use_pdflatex:
            # use pdflatex
            output += call(['pdflatex', 'document.tex'])
            output += call(['pdflatex', 'document.tex'])

        else:
            logging.getLogger(__name__).warn('Using tex->dvi->ps->pdf workflow '
                                             'might not work in newer latex')
            # use ordinary latex
            output += call(['latex', 'document.tex'])
            output += call(['latex', 'document.tex'])
            output += call(['dvips', 'document.dvi'])
            output += call(['ps2pdf', 'document.ps'])

    # output resulting PDF
    pdf_file = os.path.join(tmp, 'document.pdf')
    output += ('Copy result to `%s`\n' % outfile).encode('utf-8')
    shutil.copyfile(pdf_file, outfile)

    # house keeping
    output += ('Clear folder `%s`\n' % tmp).encode('utf-8')
    output += call(['rm', '-rf', tmp])
    
    return output
