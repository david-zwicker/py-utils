"""
Python module defining classes for creating more versatile figures

This code and information is provided 'as is' without warranty of any kind,
either express or implied, including, but not limited to, the implied
warranties of non-infringement, merchantability or fitness for a particular
purpose.
"""

from __future__ import division

import itertools
import logging
import os
import subprocess
from contextlib import contextmanager

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from six.moves import range

from . import style



GOLDEN_MEAN = 2/(np.sqrt(5) - 1)
INCHES_PER_PT = 1.0/72.27  # Convert pt to inch



def crop_pdf_file(file_input, file_output=None, silence_output=True):
    """ crops the pdf in the input file using the tool `pdfcrop`. If
    `file_output` is not given, the input file is overwritten. `silence_output`
    determines whether the output of the program is shown on the standard
    output
    """
    if file_output is None:
        file_output = file_input 
    
    cmd = ['pdfcrop', str(file_input), str(file_output)]
    
    if silence_output:
        try:
            from subprocess import DEVNULL  # py3k
        except ImportError:
            DEVNULL = open(os.devnull, 'wb')
        subprocess.check_call(cmd, stdout=DEVNULL, stderr=subprocess.STDOUT)
        
    else:
        subprocess.check_call(cmd)



class FigureBase(Figure):
    """ Extended version of a matplotlib figure """

    # factors used to determine the preferred number of ticks in each direction
    xtick_factor = 0.66
    ytick_factor = 1.

    def __init__(self, **kwargs):
        self.backend_old = None

        # read parameters
        self.transparent = kwargs.pop('transparent', None)
        self.verbose = kwargs.pop('verbose', False)
        self._logger = logging.getLogger(__name__)

        backend = kwargs.pop('backend', None)
        safe_colors = kwargs.pop('safe_colors', False)
        fig_width_pt = kwargs.pop('fig_width_pt', 246.)
        aspect = kwargs.pop('aspect', None)
        dx = kwargs.pop('dx', None)
        dy = kwargs.pop('dy', None)
        num_ticks = kwargs.pop('num_ticks', None)

        # switch backend, if requested
        if backend is not None:
            self.backend_old = plt.get_backend()
            plt.switch_backend(backend)
            if backend.lower() != plt.get_backend().lower():
                self._logger.warning('Backend could not be switched from `%s` '
                                     'to `%s`', plt.get_backend(), backend)

        # choose the color list used in this figure
        if safe_colors:
            self.colors = style.COLOR_LIST_SAFE
        else:
            self.colors = style.COLOR_LIST

        # set the number of ticks
        if num_ticks:
            if hasattr(num_ticks, '__iter__'):
                self.num_ticks = num_ticks[:2]
            else:
                self.num_ticks = (num_ticks, num_ticks)
        else:
            self.num_ticks = (None, None)

        # calculate the figure size
        if fig_width_pt is None:
            figsize = kwargs.pop('figsize', None)

        else:
            # calculate lengths
            if aspect is None:
                aspect = GOLDEN_MEAN

            if dx is None:
                dx = 4. * plt.rcParams['axes.labelsize'] / fig_width_pt
            if not hasattr(dx, '__iter__'):
                dx = (dx, 0.05)

            if dy is None:
                dy = 4.5 * plt.rcParams['axes.labelsize'] / fig_width_pt
            if not hasattr(dy, '__iter__'):
                dy = (dy, 0.05)

            if self.verbose:
                self._logger.info('Parameter dx: %g, %g', *dx)
                self._logger.info('Parameter dy: %g, %g', *dy)

            fig_width = fig_width_pt*INCHES_PER_PT  # width in inches
            axes_width = fig_width*(1. - dx[0] - dx[1])
            axes_height = axes_width/aspect    # height in inches
            fig_height = axes_height/(1. - dy[0] - dy[1])
            figsize = [fig_width, fig_height]
            kwargs.pop('figsize', None)

        # setup the figure using the inherited constructor
        super(FigureBase, self).__init__(figsize=figsize, **kwargs)

        # setup the axes using the calculated dimensions
        self.ax = self.add_axes([dx[0], dy[0], 1 - dx[0] - dx[1],
                                 1 - dy[0] - dy[1]])


    def get_color_iter(self, color=None):
        """
        Transforms the given color into a cycle or returns default style.
        """
        if color is None:
            color = self.colors

        try:
            color_iter = itertools.cycle(color)
        except TypeError:
            color_iter = itertools.repeat(color)

        return color_iter


    def get_style_iter(self, color=True, dashes=None, extra=None):
        """
        Returns an iterator of various parameters controlling the style
        of plots.
        """

        # prepare the data
        if color in [True, None]:
            icolor = itertools.cycle(self.colors)
        elif color is False:
            icolor = itertools.repeat('k')
        else:
            icolor = itertools.cycle(color)

        if dashes in [False, None]:
            idashes = itertools.repeat('-')
        elif dashes is True:
            idashes = itertools.cycle(['-', '--', ':', '-.'])
        else:
            idashes = itertools.cycle(dashes)

        # function yielding the iterator
        def _style_generator():
            """ helper function """
            while True:
                res = {'color': next(icolor)}
                if dashes is not None:
                    res['linestyle'] = next(idashes)
                if extra is not None:
                    res.update(extra)
                yield res

        return _style_generator()


    def invert_colors(self):
        """ Changes the colors of a figure to their inverted values """

        # keep track of the object that have been changed
        visited = set()

        def get_filter(name):
            """ construct a specific filter for `findobj` """
            return lambda x: (hasattr(x, 'set_%s' % name) and 
                              hasattr(x, 'get_%s' % name))

        for o in self.findobj(get_filter('facecolor')):
            if o not in visited:
                o.set_facecolor(style.invert_color(o.get_facecolor()))
                if hasattr(o, 'set_edgecolor') and hasattr(o, 'get_edgecolor'):
                    o.set_edgecolor(style.invert_color(o.get_edgecolor()))
                visited.add(o)

        for o in self.findobj(get_filter('color')):
            if o not in visited:
                o.set_color(style.invert_color(o.get_color()))
                visited.add(o)

        # update canvas
        self.canvas.draw()


    @contextmanager
    def inverted_colors(self):
        """ Invert the colors and change them back after yielding """
        self.invert_colors()
        yield self
        self.invert_colors()


    def post_process(self, legend_frame=False):
        """
        Post process the image to adjust some things
        """

        # iterate over all axes
        for ax in self.axes:
            # adjust the ticks of the x-axis
            if self.num_ticks[0] is not None:
                num_ticks_x = self.num_ticks[0]
            else:
                num_ticks_x = self.xtick_factor * \
                              plt.rcParams['figure.figsize'][0]
            style.set_presentation_style_of_axis(ax.get_xaxis(),
                                                 int(num_ticks_x))

            # adjust the ticks of the y-axis
            if self.num_ticks[1] is not None:
                num_ticks_y = self.num_ticks[1]
            else:
                num_ticks_y = self.ytick_factor * \
                              plt.rcParams['figure.figsize'][1]
            style.set_presentation_style_of_axis(ax.get_yaxis(),
                                                 int(num_ticks_y))

            # adjust the legend
            legend = ax.get_legend()
            if legend is not None:
                legend.draw_frame(legend_frame)


    def savefig_pdf(self, filename, crop_pdf=False, **kwargs):
        """
        Saves a figure as a PDF file. If the filename ends with .eps or .ps,
        we first create a EPS or PS file and then convert it to PDF.
        """

        # prepare data
        filename, extension = os.path.splitext(filename)
        if extension == '':
            extension = '.pdf'
        file_pdf = filename + '.pdf'

        # save figure in the requested format
        self.savefig(filename + extension, transparent=self.transparent,
                     **kwargs)
        
        if extension == '.ps':
            subprocess.check_call(['ps2pdf', filename + '.ps', file_pdf])
        elif extension == '.eps':
            subprocess.check_call(['epspdf', filename + '.eps', file_pdf])

        if crop_pdf:
            crop_pdf_file(file_pdf)

        return file_pdf


    def savefig_inverted(self, filename, background_facecolor=None,
                         background_edgecolor=None, **kwargs):
        """ Saves the figure to `filename` with inverted colors """

        rgb = style.get_color_converter().to_rgb

        if background_facecolor is None:
            bg_face = self.get_facecolor()
            if rgb(bg_face) == rgb(mpl.rcParamsDefault['figure.facecolor']):
                bg_face = 'k'
            else:
                bg_face = style.invert_color(bg_face)
        else:
            bg_face = background_facecolor

        if background_edgecolor is None:
            bg_edge = self.get_edgecolor()
            if rgb(bg_edge) == rgb(mpl.rcParamsDefault['figure.edgecolor']):
                bg_edge = 'none'
            else:
                bg_edge = style.invert_color(bg_edge)
        else:
            bg_edge = background_edgecolor

        # save inverted figure
        with self.inverted_colors():
            file_pdf = self.savefig_pdf(
                filename, facecolor=bg_face, edgecolor=bg_edge, **kwargs
            )
            
        return file_pdf



class FigureLatex(FigureBase):
    r""" Creates a latex figure of a certain width which should fit nicely
      The width must be given in pt and may be retrieved by
      \showthe\columnwidth or \the\columnwidth
    """

    def __init__(self, **kwargs):

        # read configuration
        font_size = kwargs.pop('font_size', 11)

        # setup all parameters
        plt.rcParams.update({
            'axes.labelsize': font_size,
            'font.family': 'serif',
            'font.size': font_size,
            'legend.fontsize': font_size,
            'xtick.labelsize': 0.9*font_size,
            'ytick.labelsize': 0.9*font_size,
            'text.usetex': True,
            'legend.loc': 'best',
            'font.serif': 'Computer Modern Roman, Times, Palatino, New Century '
                          'Schoolbook, Bookman',
            'pdf.compression': 4,
        })

        # create figure using the inherited constructor
        super(FigureLatex, self).__init__(**kwargs)



class FigurePresentation(FigureBase):
    """ Creates a figure suitable for presentations
    """

    def __init__(self, **kwargs):

        # read configuration
        font_size = kwargs.pop('font_size', 11)

        # setup latex preamble
        preamble = \
            mpl.rcsetup.validate_stringlist(plt.rcParams['text.latex.preamble'])
        if r'\usepackage{sfmath}' not in preamble:
            preamble += [
                r'\sffamily',
                r'\usepackage{sfmath}',
                r'\renewcommand{\familydefault}{\sfdefault}'
            ]

        # setup all parameters
        plt.rcParams.update({
            'axes.labelsize': font_size,
            'font.family': 'sans-serif',
            'font.size': font_size,
            'legend.fontsize': font_size,
            'xtick.labelsize': 0.9*font_size,
            'ytick.labelsize': 0.9*font_size,
            'text.usetex': True,
            'text.latex.preamble': preamble,
            'legend.loc': 'best',
            'font.sans-serif': 'Computer Modern Sans Serif, '
                'Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, '
                'Arial, Helvetica, Avant Garde, sans-serif',
            'pdf.compression': 4,
        })

        # create figure using the inherited constructor
        super(FigurePresentation, self).__init__(**kwargs)



@contextmanager
def figure_display(FigureClass=None, post_process=True, legend_frame=False,
                   **kwargs):
    """ Provides a context manager for handling figures for display """

    if FigureClass is None:
        FigureClass = FigurePresentation

    # create figure
    fig = plt.figure(FigureClass=FigureClass, **kwargs)

    # return figure for plotting
    yield fig

    # show the figure
    if post_process:
        fig.post_process(legend_frame)
    plt.show()



@contextmanager
def figure_file(filename, FigureClass=None, crop_pdf=None, post_process=True, 
                legend_frame=False, hold_figure=False, **kwargs):
    """ Provides a context manager for handling figures for latex """

    if FigureClass is None:
        FigureClass = FigurePresentation

    # create figure
    fig = plt.figure(FigureClass=FigureClass, **kwargs)

    # return figure for plotting
    yield fig

    # save the figure to a file
    if post_process:
        fig.post_process(legend_frame)
    fig.savefig_pdf(filename=filename, crop_pdf=crop_pdf)
    if not hold_figure:
        plt.close(fig)



def figures(filename, **kwargs):
    """ Generator yielding two figure instances producing a latex and a
    presentation representation of a plot """

    # split filename to be able to insert content
    name, extension = os.path.splitext(filename)

    data = (('_latex', FigureLatex), ('_presentation', FigurePresentation))
    for style, cls in data:
        filename = name + style + extension
        with figure_file(filename, cls, **kwargs) as f:
            yield f
            


def axes_broken_y(axes, upper_frac=0.5, break_frac=0.02, ybounds=None,
                  xlabel=None, ylabel=None):
    """
    Replace the current axes with a set of upper and lower axes.

    The new axes will be transparent, with a breakmark drawn between them.
    They share the x-axis.  Returns (upper_axes, lower_axes).

    If ybounds=[ymin_lower, ymax_lower, ymin_upper, ymax_upper] is defined,
    upper_frac will be ignored, and the y-axis bounds will be fixed with the
    specified values.
    """
    def breakmarks(axes, y_min, y_max, xwidth=0.008):
        x1, _, x2, _ = axes.get_position().get_points().flatten().tolist()
        segment_height = (y_max - y_min) / 3.
        xoffsets = [0, +xwidth, -xwidth, 0]
        yvalues = [y_min + (i * segment_height) for i in range(4)]
        # Get color of y-axis
        for loc, spine in axes.spines.items():
            if loc == 'left':
                color = spine.get_edgecolor()
        for x_position in [x1, x2]:
            line = mpl.lines.Line2D(
                [x_position + offset for offset in xoffsets], yvalues,
                transform=plt.gcf().transFigure, clip_on=False,
                color=color)
            axes.add_line(line)
    # Readjust upper_frac if ybounds are defined
    if ybounds:
        if len(ybounds) != 4:
            print("len(ybounds) != 4; aborting...")
            return
        ymin1, ymax1, ymin2, ymax2 = [float(value) for value in ybounds]
        data_height1, data_height2 = (ymax1 - ymin1), (ymax2 - ymin2)
        upper_frac = data_height2 / (data_height1 + data_height2)
    x1, y1, x2, y2 = axes.get_position().get_points().flatten().tolist()
    width = x2 - x1
    lower_height = (y2 - y1) * ((1 - upper_frac) - 0.5 * break_frac)
    upper_height = (y2 - y1) * (upper_frac - 0.5 * break_frac)
    upper_bottom = (y2 - y1) - upper_height + y1
    lower_axes = plt.axes([x1, y1, width, lower_height], axisbg='None')
    upper_axes = plt.axes([x1, upper_bottom, width, upper_height],
                          axisbg='None', sharex=lower_axes)
    # Erase the edges between the axes
    for loc, spine in upper_axes.spines.items():
        if loc == 'bottom':
            spine.set_color('none')
    for loc, spine in lower_axes.spines.items():
        if loc == 'top':
            spine.set_color('none')
            
    upper_axes.get_xaxis().set_ticks_position('top')
    lower_axes.get_xaxis().set_ticks_position('bottom')
    plt.setp(upper_axes.get_xticklabels(), visible=False)
    breakmarks(upper_axes, y1 + lower_height, upper_bottom)
    
    # Set ylims if ybounds are defined
    if ybounds:
        lower_axes.set_ylim(ymin1, ymax1)
        upper_axes.set_ylim(ymin2, ymax2)
        lower_axes.set_autoscaley_on(False)
        upper_axes.set_autoscaley_on(False)
        
        label_pos_upper = (0, 1 - (0.5 / (upper_frac / (1 + break_frac))))
        upper_axes.yaxis.get_label().set_position(label_pos_upper)
        label_pos_lower = (0, 0.5 / ((1 - upper_frac) / (1 + break_frac)))
        lower_axes.yaxis.get_label().set_position(label_pos_lower)
        
    # Make original axes invisible
    axes.set_xticks([])
    axes.set_yticks([])
    for loc, spine in axes.spines.items():
        spine.set_color('none')
    return upper_axes, lower_axes



if __name__ == "__main__":
    print('This file is intended to be used as a module.')
    print('This code serves as a test for the defined methods.')

    test_x = np.linspace(0, 10, 100)

    # test plot with presentation style
    for fig in figures('test.pdf'):
        plt.plot(test_x, np.sin(test_x), "r", test_x, np.cos(test_x), "b")
        plt.xlabel("Coordinate $x$")
        plt.ylabel("f(x)")
        plt.title("Simple Plot")
        plt.legend(("sin(x)", "cos(x)"))
        
        
    # test these
    ax = plt.axes()
    upper, lower = axes_broken_y(ax, ybounds=[-2., 2.9, 22.1, 30.])
    upper.plot(range(30), range(30))
    lower.plot(range(30), range(30))
    upper.set_ylabel('Data')
    plt.show()
        
