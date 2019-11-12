'''
Created on Aug 9, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
import six

from ..link.latex import number2latex


# nice colors
COLOR_BLUE_OLD = '#0673B7'
COLOR_ORANGE_OLD = '#FF7600'
COLOR_GREEN_OLD = '#00A919'
COLOR_RED_OLD = '#E6001C'

# colors suitable for color blind people
COLOR_BLUE_SAFE = '#0673B7'
COLOR_ORANGE_SAFE = '#EFE342'
COLOR_GREEN_SAFE = '#009D73'
COLOR_RED_SAFE = '#D45F14'

COLOR_BLUE = '#0673B7'
COLOR_ORANGE = '#FF7600'
COLOR_GREEN = '#00A919'
COLOR_RED = '#E6001C'

# this list has been taken from
# Wong. Color blindness. Nat Methods (2011) vol. 8 (6) pp. 441
COLOR_LIST_SAFE = [
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#009E73',  # Bluish green
    '#E69F00',  # Orange
    '#56B4E9',  # Sky blue
    '#F0E442',  # Yellow
    '#CC79A7',  # Reddish purple
    'k'         # Black
]
COLOR_LIST = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED, 'k']
COLOR_LIST_STANDARD = ['b', 'g', 'r', 'c', 'm', 'y', 'k']



def get_color_iter(color=None):
    """
    Transforms the given color into a cycle or returns default colors
    """
    if color is None:
        color = COLOR_LIST

    try:
        color_iter = itertools.cycle(color)
    except TypeError:
        color_iter = itertools.repeat(color)

    return color_iter


plot_colors = get_color_iter



def get_style_iter(color=True, dashes=None, extra=None):
    """ Returns an iterator of various parameters controlling the style
    of plots """

    # prepare the data
    if color in [True, None]:
        icolor = itertools.cycle(COLOR_LIST)
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
        """ Helper function """
        while True:
            res = {'color': next(icolor)}
            if dashes is not None:
                res['linestyle'] = next(idashes)
            if extra is not None:
                res.update(extra)
            yield res

    return _style_generator()


plot_styles = get_style_iter



def get_colormap(colors, name=None):
    """ builds a segmented colormap with the color sequence given as a string,
    e.g., 'rgb' for a colormap cycling through red, green, and blue.
    Alternatively, a list of colors can be given.
    
    `name` sets the internal name of the colormap
    """
    if isinstance(colors, six.string_types):
        COLORS = {
            'r': '#E6001C', 'g': '#00A919', 'b': '#0673B7',
            'w': '#FFFFFF', 'k': '#000000'
        }
        colors = [COLORS[c] for c in colors]
    
        if name is None:
            name = colors
            
    elif name is None:
        name = 'custom'

    return mclr.LinearSegmentedColormap.from_list(name, colors) 
    


def blend_colors(color, bg='w', alpha=0.5):
    """
    Blends two colors using a weight. Can be used for faking alpha
    blending
    """
    if not hasattr(blend_colors, 'to_rgb'):
        blend_colors.to_rgb = mclr.ColorConverter().to_rgb

    return alpha*np.asarray(blend_colors.to_rgb(color)) \
        + (1 - alpha)*np.asarray(blend_colors.to_rgb(bg))



def set_axis_color(ax=None, axis='both', color='r'):
    """
    Changes the color of an axis including the ticks and the label
    
    `ax` determines what axes object to use. If `None`, use current axes
    `axis` determines which axes will be changed. Allowed values are any of
        ['x', 'y', 'both']
    `color` sets the actual color
    """
    # determine axes object
    if ax is None:
        ax = plt.gca()

    # determine which axes should be affected
    if axis == 'both':
        axes = ['x', 'y']
    elif axis == 'x' or axis == 'y':
        axes = [axis]
    else:
        raise ValueError("`axis` parameter must be from ['x', 'y', 'both']")
    
    # change the color of the requested axes
    for axis in axes:
        ax.tick_params(axis=axis, which='both', color=color, labelcolor=color)



def get_color_scheme(base_color, num=4, spread=1.):
    """ Distributes num colors around the color wheel starting with a base
    color and converting the fraction `spread` of the circle """
    base_rgb = mclr.colorConverter.to_rgb(base_color)
    base_rgb = np.reshape(np.array(base_rgb), (1, 1, 3))
    base_hsv = mclr.rgb_to_hsv(base_rgb)[0, 0]
    res_hsv = np.array([[
        ((base_hsv[0] + dh) % 1., base_hsv[1], base_hsv[2])
        for dh in np.linspace(-0.5*spread, 0.5*spread, num, endpoint=False)
    ]])
    return mclr.hsv_to_rgb(res_hsv)[0]



def get_color_converter():
    """ returns a (cached) matplotlib color converter """ 
    # initialize the color converter if necessary
    if not hasattr(get_color_converter, '_color_converter'):
        get_color_converter._color_converter = mclr.ColorConverter()
    return get_color_converter._color_converter



def invert_color(color):
    """ Returns the inverted value of a matplotlib color """
    # get the color value
    c = get_color_converter().to_rgba(color)
    # keep alpha value intact!
    return (1-c[0], 1-c[1], 1-c[2], c[3])



def set_presentation_style_of_axis(axis, num_ticks=7, use_tex=True):
    """ private function setting a single axis to presentation style """

    # adjust tick formatter
    if use_tex:
        def apply_format(val, val_str):
            """ Helper function applying the format """
            return number2latex(val, add_dollar=True)
        axis.set_major_formatter(FuncFormatter(apply_format))
        if axis.get_scale() == 'linear':
            axis.set_minor_formatter(FuncFormatter(apply_format))
    else:
        axis.set_major_formatter(FormatStrFormatter('%g'))
        if axis.get_scale() == 'linear':
            axis.set_minor_formatter(FormatStrFormatter('%g'))

    # adjust the number of ticks
    if num_ticks == 0:
        axis.set_ticks([])
    elif axis.get_scale() == 'linear':
        axis.set_major_locator(MaxNLocator(num_ticks, steps=[1, 2, 5, 10]))



def set_presentation_style(fig=None, legend_frame=False, axes=None, scale=1.):
    """ Changes the style of a figure to be useful for presentations """

    # secondary scale factor
    scale2 = np.sqrt(scale)

    # get the right figure handle
    if fig is None:
        fig = plt.gcf()

    # compile the list of axes
    if axes is None:
        axes = set()
    else:
        axes = set(axes)
    axes |= set(fig.axes)

    # run through all axes objects
    for ax in axes:

        # adjust the axes
        set_presentation_style_of_axis(ax.get_xaxis())
        set_presentation_style_of_axis(ax.get_yaxis())

        # adjust the frame around the image
        for spine in ax.spines.values():
            spine.set_linewidth(2.*scale)

        # adjust all lines within the axes
        for line in ax.get_lines():
            line.set_linewidth(3./2.*scale*line.get_linewidth())

        # adjust all text objects
        for text in ax.findobj(plt.Text):
            text.set_fontname('Helvetica')
            text.set_fontsize(16*scale2)

        # adjust the tick padding
        ax.tick_params(
            pad=6*scale, width=2*scale, length=7*scale
        )

        # adjust the legend, if there is any
        legend = ax.get_legend()
        if legend is not None:
            legend.draw_frame(legend_frame)
            for line in legend.get_lines():
                line.set_linewidth(2.*scale)


    # redraw figure
    fig.canvas.draw()



def reordered_legend(order=None, ax=None, *args, **kwargs):
    """
    Reorders the legend of an axis
    """
    if ax is None:
        ax = plt.gca()

    ax.legend(*args, **kwargs)
    if order is not None:
        handles, labels = ax.get_legend_handles_labels()

        handles = np.asarray(handles)
        labels = np.asarray(labels)

        ax.legend(handles[order], labels[order], *args, **kwargs)



if __name__ == "__main__":
    print('This file is intended to be used as a module.')
    print('This code serves as a test for the defined methods.')

    tests = (
        'safe_colors',
        'axis_color',
        'presentation_style',
    )
    

    if 'safe_colors' in tests:
        plt.clf()

        for i, color in enumerate(COLOR_LIST_SAFE):
            plt.axhspan(0.5-i, -i, color=color)

        plt.show()


    if 'axis_color' in tests:
        plt.clf()

        test_x = np.linspace(0, 5, 100)
        # testplot with colorful axis

        plt.plot(test_x, np.sin(test_x), c='r')
        plt.xlabel("x")
        plt.ylabel("sin(x)")

        set_axis_color(axis='x', color='b')
        set_axis_color(axis='y', color='r')

        ax2 = plt.twinx()
        plt.plot(test_x, np.cos(test_x), c='g')
        plt.ylabel("cos(x)")
        set_axis_color(ax2, axis='y', color='g')

        plt.show()


    if 'presentation_style' in tests:
        test_x = np.linspace(0, 5, 100)

        # testplot with presentation style
        plt.clf()
        plt.plot(test_x, np.sin(test_x), "r", test_x, np.cos(test_x), "b")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Simple Plot")
        plt.legend(("sin(x)", "cos(x)"))

        # apply presentation style to plot
        set_presentation_style()

        plt.show()
        