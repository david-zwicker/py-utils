'''
Created on Aug 9, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import Locator
import six
from six.moves import range



def doublearrow(xs, ys, w=0.1, **kwargs):
    """ Plots a double arrow between the two given coordinates """

    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    # set parameters of the arrow
    arrowparams = {
        'head_width': 2*w,
        'head_length': w,
        'length_includes_head': True,
        'shape': 'full',
        'head_starts_at_zero': False
    }
    arrowparams.update(kwargs)

    # plot two arrows to mimic double arrow
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    ax.arrow(xs[0], ys[0], dx, dy, **arrowparams)
    ax.arrow(xs[1], ys[1], -dx, -dy, **arrowparams)
    
    

def log_slope_indicator(xmin=1., xmax=2., factor=None, ymax=None, exponent=1.,
                        label_x='', label_y='', space=15, loc='lower', ax=None,
                        debug=False, **kwargs):
    """
    Function adding a triangle to axes `ax`. This is useful for indicating
    slopes in log-log-plots. `xmin` and `xmax` denote the x-extend of the
    triangle. The y-coordinates are calculated according to the formula
        y = factor*x**exponent
    If supplied, the texts `label_x` and `label_y` are put next to the
    catheti. The parameter `loc` determines whether the catheti are
    above or below the diagonal. Additionally, kwargs can be used to
    set the style of the triangle
    
    `loc` determines whether the triangle appears above (`loc='upper'`) or below
        (`loc='lower'; default) the diagonal line.
    """

    # prepare the axes and determine 
    if ax is None:
        ax = plt.gca()
        
    if loc == 'lower':
        lower = (exponent > 0)
    elif loc == 'upper':
        lower = (exponent < 0)
    else:
        raise ValueError('`loc` must be either `lower` or `upper`.')

    if ymax is not None:
        factor = ymax/max(xmin**exponent, xmax**exponent)

    if factor is None:
        factor = 1.

    # get triangle coordinates
    y = factor*np.array((xmin, xmax), np.double)**exponent
    if lower:
        pts = np.array([[xmin, y[0]], [xmax, y[0]], [xmax, y[1]]])
    else:
        pts = np.array([[xmin, y[0]], [xmax, y[1]], [xmin, y[1]]])

    if debug:
        print('The coordinates of the log slope indicator are %s' % pts)

    # add triangle to axis
    if not('facecolor' in kwargs or 'fc' in kwargs):
        kwargs['facecolor'] = 'none'
    if not('edgecolor' in kwargs or 'ec' in kwargs):
        kwargs['edgecolor'] = 'k'
    p = Polygon(pts, closed=True, **kwargs)
    ax.add_patch(p)

    # add labels
    xt = np.exp(0.5*(np.log(xmin) + np.log(xmax)))
    # dx = (xmax/xmin)**0.1
    yt = np.exp(np.log(y).mean())
    # dy = (y[1]/y[0])**0.1
    sgn = np.sign(exponent)
    if lower:
        ax.annotate(
            label_x, xy=(xt, y[0]), xytext=(0, -sgn*space),
            textcoords='offset points', size='x-small',
            horizontalalignment='center',
            verticalalignment='top'
        )
        ax.annotate(
            label_y, xy=(xmax, yt), xytext=(space, 0),
            textcoords='offset points', size='x-small',
            horizontalalignment='right',
            verticalalignment='center'
        )

    else:
        ax.annotate(
            label_x, xy=(xt, y[1]), xytext=(0, sgn*space),
            textcoords='offset points', size='x-small',
            horizontalalignment='center',
            verticalalignment='bottom'
        )
        ax.annotate(
            label_y, xy=(xmin, yt), xytext=(-space, 0),
            textcoords='offset points', size='x-small',
            horizontalalignment='left',
            verticalalignment='center'
        )



class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))



def render_table(data, col_width=3.0, row_height=0.625, font_size=14,
                 header_color='#40466e', row_colors=['#f1f1f2', 'w'],
                 edge_color='w', bbox=[0, 0, 1, 1], header_columns=0,
                 ax=None, **kwargs):
    """
    Renders the table given in `data` in a matplotlib axes.
    
    Code inspired by http://stackoverflow.com/a/39358722/932593
    """
    
    if ax is None:
        size = ((np.array(data.shape[::-1]) + np.array([0, 1])) *
                np.array([col_width, row_height]))
        _, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox,
                         colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
            
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax



def determine_label_positions(pos, sigma=0.05, repulsion=0.1, attraction=0.1,
                              steps=100, noise=1e-3):
    """ determines label positions automatically by moving labels that are too
    close a bit apart. The algorithm is based on a physical system with labels
    connected to the given position by springs of stiffness `attraction`, while
    all other labels possess repulsive potentials of strength `repulsion` and
    range `sigma`. The physical system is solved by iterating `step` times and
    we additionally put noise of strength `noise` to break some degenerate
    situations.
    
    Note that this function assumes that labels are positioned on a linear
    scale. If labels should be positioned on a log-scale, the positions should
    be transformed to a linear scale before and after applying this function.    
    """
    pos = np.array(pos, dtype=np.double)  # turn into array and make a copy
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError('Input data does not seem to be a 2d coordinate list')
    dim = len(pos)
    
    # scale positions to unity
    pos_mean = pos.mean(axis=0)
    pos -= pos_mean
    pos_scale = np.abs(pos).max(axis=0)
    pos /= pos_scale

    pos_orig = pos.copy()
    
    # iterate several times to find a good position
    for _ in range(steps):
        # apply noise term
        pos += noise * (np.random.random(dim)[:, None] - 0.5)
        
        # iterate over all positions
        for i in range(dim):
            # evaluate distance to original position
            pos[i] -= attraction * (pos[i] - pos_orig[i])

            # evaluate distance to all other positions
            diff = pos[i] - pos
            dist = np.linalg.norm(diff, axis=1)
            j = (dist != 0)
            force = diff[j] / dist[j, None] * np.exp(-(dist[j, None]/sigma)**2)
            pos[i] += repulsion * np.sum(force, axis=0)
    
    return pos * pos_scale + pos_mean



def add_scaled_colorbar(im, ax, aspect=20, pad_fraction=0.5, **kwargs):
    """ add a vertical color bar to an image plot
    
    The height of the colorbar is now adjusted to the plot, so that the width
    determined by `aspect` is now given relative to the height. Moreover, the
    gap between the colorbar and the plot is now given in units of the fraction
    of the width by `pad_fraction`. 
    
    Inspired by https://stackoverflow.com/a/33505522/932593
    """
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)            
       


if __name__ == "__main__":
    print('This file is intended to be used as a module.')
    print('This code serves as a test for the defined methods.')

    tests = (
        'log_slope_indicator',
        'log_slope_indicator_neg',
    )

    if 'log_slope_indicator' in tests:
        test_x = np.logspace(0, 3, 20)
        test_y = test_x**2
        test_y *= (1 + 0.1*np.random.randn(20))

        plt.loglog(test_x, test_y, '+')

        log_slope_indicator(
            xmin=10, xmax=100, factor=0.5, exponent=2.,
            label_x='1', label_y='2', ec='red'
        )
        log_slope_indicator(
            xmin=100, xmax=300, factor=2., exponent=2.,
            label_x='1', label_y='2', loc='upper'
        )

        plt.show()

    if 'log_slope_indicator_neg' in tests:
        test_x = np.logspace(0, 3, 20)
        test_y = test_x**-2
        test_y *= (1 + 0.1*np.random.randn(20))

        plt.loglog(test_x, test_y, '+')

        log_slope_indicator(
            xmin=10, xmax=100, factor=0.5, exponent=-2.,
            label_x='1', label_y='2', ec='red'
        )
        log_slope_indicator(
            xmin=100, xmax=300, factor=2., exponent=-2.,
            label_x='1', label_y='2', loc='upper'
        )

        plt.show()
        