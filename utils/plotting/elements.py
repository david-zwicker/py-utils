'''
Created on Aug 9, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import Locator
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
    """

    # prepare the axes and determine 
    if ax is None:
        ax = plt.gca()
    lower = (loc == 'lower')

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



if __name__ == "__main__":
    print('This file is intended to be used as a module.')
    print('This code serves as a test for the defined methods.')

    from ..math import logspace

    tests = (
        'log_slope_indicator',
    )

    if 'log_slope_indicator' in tests:
        test_x = logspace(1, 1000, 20)
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
        