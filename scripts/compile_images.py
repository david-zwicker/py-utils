#!/usr/bin/env python
"""
Small script which combines images given on the command line into a single
PDF file.

Basic Usage:
    ./compile_images -o output.pdf input1.eps input2.eps

Run `./compile_images --help` for more details.

@copyright: David Zwicker <dzwicker@seas.harvard.edu>
@date:      2014-07-10
"""

from __future__ import division

import os
from optparse import OptionParser

from utils.link import latex



def get_options():
    """ parse the command line options """
    
    # define options of the command line
    parser = OptionParser("compile_images.py [options] image1 [image2]...")
    parser.add_option(
        "-2", "--twocolumns", action="store_true", dest="twocolumns",
        default=False, help="use two columns for images"
    )
    parser.add_option(
        "-c", "--caption", dest="caption", help="add a caption to the PDF",
        metavar="TITLE"
    )
    parser.add_option(
        "-o", "--output", dest="output", default='out.pdf',
        help="write resulting PDF to FILE", metavar="FILE"
    )
    parser.add_option(
        "-i", "--information", dest="infofile",
        help="includes TeX-code from FILE", metavar="FILE"
    )
    parser.add_option(
        "-t", "--tex", action="store_true", dest="tex", default=False,
        help="output TEX source code and do not compile"
    )
    parser.add_option(
        "-n", "--no-title", action="store_true", dest="no_title", default=False,
        help="prevents the addition of a title below the image"
    )
    parser.add_option(
        "-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="print status messages to stdout"
    )
    
    # parse options from the command line
    options, args = parser.parse_args()
    
    # display help if no files are given
    if len(args) == 0:
        parser.print_help()
        exit()
        
    return options, args



def main():
    """ main program """
    options, args = get_options()
    
    # set some initial parameters depending on the parsed options
    if options.output is None:
        outfile = None
        options.verbose = False
    else:
        outfile = os.path.abspath(options.output)
    
    if options.twocolumns:
        title_characters = 45
        width_string = "0.45\\textwidth"
    else:
        title_characters = 80
        width_string = "\\textwidth"
        
    # create TeX code
    if options.verbose:
        print("Scanning images...")
    
    # check whether a caption is requested
    if options.caption is None:
        tex_source = ""
    else:
        tex_source = ("\\begin{center}\n"
                      "\\section*{%s}\n"
                      "\\end{center}\n"
                      "\\thispagestyle{empty}\n" % options.caption)
    
    # add possible extra information
    if options.infofile is not None:
        file_handle = open(options.infofile, 'r')
        tex_source += ('\\begin{minipage}{\textwidth}\n' +
                       file_handle.read() + '\n' +
                       '\\end{minipage}\n\\')
        file_handle.close()
    
    # initialize data
    use_pdflatex, use_latex = False, False
    
    # run through all remaining options, which are the filenames of the images
    for k, filename in enumerate(args):
    
        # check the image file
        img = os.path.abspath(filename)
        if os.path.exists(img):
            title = img
            filename = img
            ext = os.path.splitext(img)[1]
            if ext == ".eps":
                use_latex = True
            elif ext == ".pdf":
                use_pdflatex = True
            else:
                use_pdflatex = True
                
        elif os.path.exists(img + ".eps"):
            title = img
            filename = "%s.eps" % img
            use_latex = True
            
        elif os.path.exists(img + ".pdf"):
            title = img
            filename = "%s.pdf" % img
            use_pdflatex = True
            
        else:
            if options.verbose:
                print("File '%s' does not exist and will be ignored" % img)
            continue
    
        # handle the title
        if options.no_title:
            title_str = ""
        else:
            # strip title string if it is too long
            if len(title) > title_characters:
                title = "...%s" % title[3-title_characters:]
            title_str = "\\verb\"%s\"" % title
    
        # add the necessary tex-code
        tex_source += ("\\begin{minipage}{%(width)s}\n"
                       "\\includegraphics[width=\\textwidth]{%(filename)s}\n"
                       "%(title)s\n"
                       "\\end{minipage}\n" % {'width': width_string,
                                              'filename': filename,
                                              'title': title_str})
    
        # handle the space between figures
        if options.twocolumns and k % 2 == 0:
            # we have to place the figure next to another one
            tex_source += "\\hfill\n"
    
    
    # define latex document structure
    tex_source = ("\\documentclass[10pt]{article}\n"
                  "\\usepackage[top=2cm, bottom=2cm, left=1cm, right=1cm, "
                               "a4paper]{geometry}\n"
                  "\\usepackage{graphicx} %% include graphics\n"
                  "\\usepackage{grffile}  %% support spaces in filenames\n"
                  "\\begin{document}%%\n" 
                  + tex_source +
                  "\\vfill\n\\end{document}\n")
    
    # decide what to output
    if options.tex:
        # output TeX code
        print(tex_source)
        
    else:
        if options.verbose:
            print("Compiling PDF file...")
    
        if use_pdflatex and use_latex:
            print("Can't compile images, since both EPS and PDF files are "
                  "given.")
            output = ""
            
        elif use_latex:
            output = latex.tex2pdf(tex_source, outfile, use_pdflatex=False) 

        else:
            output = latex.tex2pdf(tex_source, outfile, use_pdflatex=True) 
            
        if options.verbose:
            print(output)


if __name__ == '__main__':
    main()
