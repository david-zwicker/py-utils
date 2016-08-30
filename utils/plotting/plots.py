""" This file contains several functions useful for plotting figures using
matplotlib """

from __future__ import division

import itertools
import warnings

import numpy as np
from numpy.linalg import norm, eig, eigvals, solve
from scipy import interpolate, ndimage, optimize

import matplotlib.pyplot as plt
import matplotlib.colors as mclr

from ..math import logspace
from . import style



COS45 = np.cos(0.25*np.pi)



def _get_regions(values):
    """ returns an array of ids for each distinct region """
    weights = 2**np.arange(0, values.shape[2])
    return np.dot(values, weights)



def scale_image(img_in, dim_out):
    """
    Scales an image to the new_dimensions `dim_out`.
    Taken from http://stackoverflow.com/q/5586719/932593
    """

    # process input
    img_in = np.atleast_2d(img_in)
    dim_in = img_in.shape

    # setup interpolation object
    x = np.arange(0, dim_in[0])
    y = np.arange(0, dim_in[1])
    interp = interpolate.RectBivariateSpline(x, y, img_in, kx=2, ky=2)

    # calculate the new image
    xx = np.linspace(0, x.max(), dim_out[0])
    yy = np.linspace(0, y.max(), dim_out[1])
    return interp(xx, yy)



def get_contour_plot_hatched(values, colors, **kwargs):
    """
    Takes three dimensional boolean data and projects out the last dimension,
    by using hatched regions to symbolize parts, where several variables are
    true. The function returns an array with colors at each point.
    The style of the image can be influenced by these parameters:

    background_color    Color chosen, if all entries are False
    stripe_width        Width of the stripes indicating overlapping regions
    stripe_orientation  Orientation of the stripes ('\', '-', '/', or '|')

    boundary            True, if the boundary should be plotted
    boundary_width      Width of the boundary in pixel
    """

    # process general parameters
    background_color = kwargs.pop('background_color', 'w')
    nan_color = kwargs.pop('nan_color', 'k')
    return_all = kwargs.pop('return_all', False)

    # process parameters of the stripes
    stripe_width = kwargs.pop('stripe_width', 0.05)
    if stripe_width < 1:
        stripe_width = int(stripe_width*max(values.shape[:2]))
    stripe_width_diag = stripe_width/COS45
    stripe_orientation = kwargs.pop('stripe_orientation', '/')

    # process parameters of the boundary
    boundary = kwargs.pop('boundary', False)
    boundary_width = kwargs.pop('boundary_width', 1)

    # build orientation list to unify input
    dimensions = len(values[0, 0, :])
    if hasattr(stripe_orientation, '__iter__'):
        orientations = itertools.cycle(stripe_orientation)
        orientations = list(itertools.islice(orientations, dimensions))
    else:
        orientations = [stripe_orientation]*dimensions

    # convert the input and calculate the number of input values
    values = np.atleast_3d(values)
    nums = values.sum(2).astype(int)
    nan_as_int = np.array(np.nan).astype(int)

    # translate the color to RGB values
    get_color = mclr.ColorConverter().to_rgb
    background_color = get_color(background_color)
    nan_color = get_color(nan_color)
    colors = [get_color(c) for c in colors]

    # check, if the boundary values have to be calculated
    if boundary:
        # associate each region with a unique float number
        regions = _get_regions(values)
        regions[np.isnan(regions)] = -1

        # construct a filter for boundary detection
        w = np.ceil(boundary_width)
        coords = np.arange(-2*w, 2*w+1)
        x, y = np.meshgrid(coords, coords)
        r = np.sqrt(x**2 + y**2)
        sigma = 0.5*boundary_width + 0.5
        filter_matrix = 0.5 - 0.5*np.tanh(2*(r - sigma))

        # convolute the image with the filter for each region to get boundaries
        boundary_point = np.zeros(values.shape[:2])
        for k in np.unique(regions):
            panels = (regions != k)
            edges = ndimage.convolve(panels.astype(float), filter_matrix)
            boundary_point[~panels] += edges[~panels]

        boundary_point[boundary_point > 1] = 1.

    else:
        # no boundary points requested
        boundary_point = np.zeros(values.shape[:2])

    # define the different orientations of the stripe pattern
    orientation_functions = {
        '\\': lambda x, y: ((x - y) % stripe_width_diag) * nums[x, y] /
                            stripe_width_diag,
        '-': lambda x, y: (x % stripe_width)*nums[x, y]/stripe_width,
        '/': lambda x, y: ((x + y) % stripe_width_diag) * nums[x, y] / 
                            stripe_width_diag,
        '|': lambda x, y: (y % stripe_width) * nums[x, y] / stripe_width
    }

    # iterate over all pixels and process them individually
    res = np.zeros((values.shape[0], values.shape[1], 3))
    for x, y in np.ndindex(values.shape[:2]):
        if nums[x, y] == nan_as_int:
            res[x, y, :] = nan_color
        elif nums[x, y] == 0:
            res[x, y, :] = background_color
        else:
            # choose a color index based on the current stripe
            try:
                i = orientation_functions[orientations[nums[x, y] - 1]](x, y)
            except KeyError:
                raise ValueError('Allowed stripe orientation values: %s'
                                 % (', '.join(orientation_functions.keys())))

            # choose the color from the current values
            color_index = np.nonzero(values[x, y, :])[0][int(i)]
            res[x, y] = colors[color_index]

    # change color, if it is a boundary point
    res *= (1 - boundary_point[:, :, np.newaxis])

    if return_all:
        return res, kwargs
    else:
        return res


def make_contour_plot_hatched(values, colors, **kwargs):
    """
    Takes three dimensional boolean data and projects out the last dimension,
    by using hatched regions to symbolize parts, where several variables are
    true. The function uses imshow to plot the image to the current axes.
    The style of the image can be influenced by these parameters:

    background_color    Color chosen, if all entries are False
    stripe_width        Width of the stripes used in overlapping regions
    stripe_orientation  Orientation of the stripes ('\', '-', '/', or '|')

    boundary            True, if the boundary should be plotted
    boundary_color      Color of the boundary
    boundary_style      Dictionary of additional styles used in the contour
                        plot of the boundary
    boundary_smoothing  Float number determining the smoothing of the boundary

    """
    # read parameters
    boundary = kwargs.pop('boundary', True)
#     boundary_color = kwargs.pop('boundary_color', 'k')
    boundary_style = kwargs.pop('boundary_style', {})
    boundary_smoothing = kwargs.pop('boundary_smoothing', 1)

    # extract parameters used in both the imshow and the contour plot
    plot_parameters = {
        val: kwargs.pop(val, None)
        for val in ('alpha', 'origin', 'extent')
    }
    boundary_style.update(plot_parameters)

    # calculate the plot
    img, imshow_kwargs = get_contour_plot_hatched(
        values, colors, return_all=True, **kwargs
    )
    # do the plot of the image using
    imshow_kwargs.update(plot_parameters)
    plt.imshow(img, **imshow_kwargs)

    # plot boundaries if requested
    if boundary:
        regions = _get_regions(values)
        # plot a contour plot for each region
        for k in np.unique(regions):
            data = (regions == k).astype(float)
            if boundary_smoothing and boundary_smoothing > 0:
                data = ndimage.gaussian_filter(
                    data, boundary_smoothing, mode='mirror'
                )
#             contour = plt.contour(
#                 data, levels=[0.5], colors=boundary_color, **boundary_style
#             )

            # iterate over paths and simplify them
#             for c in contour.collections:
#                 for p in c.get_paths():
#                     p.simplify_threshold = .1
#                     p.vertices, p.codes = mpath.cleanup_path(
#                         p, None, True, None, False, 1., None, True
#                     )




class SteadyStateStreamPlot(object):
    """
    class that can analyze a 2D differential equation and calculated and plot
    steady states, stream lines, and similar information.
    """
    
    def __init__(self, func, region, region_constraint=None):
        """
        `func` is the vector function that defines the 2D flow field.
        The function must accept and also return an array with 2 elements
        `region` defines the region of interest for plotting. Four numbers need
            to be specified: [left, bottom, right, top].
        """
        self.func = func
        self.region = region
        if region_constraint is None:
            self.region_constraint = set()
        else:
            self.region_constraint = set(region_constraint)
        
        self.steady_states = None
        self.step = min(
            region[2] - region[0],
            region[3] - region[1]
        )/100 

        # determine region of interest
        self.rect = [
            self.region[0] - 1e-4, self.region[1] - 1e-4,
            self.region[2] + 1e-4, self.region[3] + 1e-4
        ]


    def point_in_region(self, point, strict=False):
        """
        checks whether `point` is in the region of interest
        """
        if strict:
            rect = self.region
        else:
            rect = self.rect
        return rect[0] < point[0] < rect[2] and rect[1] < point[1] < rect[3]


    def point_at_border(self, point, tol=1e-6):
        """
        returns 0 if point is not at a border, otherwise returns a
        positive number indicating which border the points belongs to
        """
        res = 0
        if np.abs(point[0] - self.region[0]) < tol:
            res += 1
        if np.abs(point[1] - self.region[1]) < tol:
            res += 2
        if np.abs(point[0] - self.region[2]) < tol:
            res += 4
        if np.abs(point[1] - self.region[3]) < tol:
            res += 8
        return res


    def jacobian(self, point, tol=1e-6):
        """
        returns the Jacobian around a point
        """
        jacobian = np.zeros((2, 2))
        jacobian[0, :] = self.func(point + np.array([tol, 0]))
        jacobian[1, :] = self.func(point + np.array([0, tol]))
        return jacobian


    def point_is_steady_state(self, point, tol=1e-6):
        """
        Checks whether `point` is a steady state
        """

        vel = self.func(point)  # velocity at the point
        x_check, y_check = False, False  # checked direction
        
        # check special boundary cases
        if ('left' in self.region_constraint and 
                np.abs(point[0] - self.region[0]) < 1e-8):
            if vel[0] > 0:
                return False
            x_check = True
            
        elif ('right' in self.region_constraint and 
                np.abs(point[0] - self.region[2]) < 1e-8):
            if vel[0] < 0:
                return False
            x_check = True 

        if ('bottom' in self.region_constraint and 
                np.abs(point[1] - self.region[1]) < 1e-8):
            if vel[1] > 0:
                return False
            y_check = True
            
        elif ('top' in self.region_constraint and 
              np.abs(point[1] - self.region[3]) < 1e-8):
            if vel[1] < 0:
                return False
            y_check = True
        
        # check the remaining directions
        if x_check and y_check:
            return True  # both x and y direction are stable
        elif x_check:
            return np.abs(vel[1]) < tol  # y direction has to be tested
        elif y_check:
            return np.abs(vel[0]) < tol  # x direction has to be tested
        else:
            return norm(vel) < tol  # both directions have to be tested
        

    def point_is_stable(self, point, tol=1e-5):
        """
        returns true if a given steady state is stable
        """
        
        if not self.point_is_steady_state(point, tol):
            raise ValueError('Supplied point is not a steady state')
        
        jacobian = self.jacobian(point, tol)
        x_check, y_check = False, False  # checked direction
        
        # check special boundary cases
        if ('left' in self.region_constraint and 
                np.abs(point[0] - self.region[0]) < 1e-8):
            x_check = True 
            
        elif ('right' in self.region_constraint and 
              np.abs(point[0] - self.region[2]) < 1e-8):
            x_check = True 

        if ('bottom' in self.region_constraint and 
                np.abs(point[1] - self.region[1]) < 1e-8):
            y_check = True
            
        elif ('top' in self.region_constraint and 
              np.abs(point[1] - self.region[3]) < 1e-8):
            y_check = True
        
        # check the remaining directions
        if x_check and y_check:
            return True  # both x and y direction are stable
        elif x_check:
            return jacobian[1, 1] < 0  # y direction has to be tested
        elif y_check:
            return jacobian[0, 0] < 0  # x direction has to be tested
        else:
            return all(eigvals(jacobian) < 0)  # both dir. have to be tested


    def get_steady_states_at_x_boundary(self, grid_points=32, loc='lower',
                                        points=None):
        """
        finds steady state points along the x-boundary at position `loc`
        """
        
        if points is None:
            points = np.array([[]])  # array that will contain all the points
            
        if loc == 'lower':
            y0 = self.region[1]
            direction = 1
        elif loc == 'upper':
            y0 = self.region[3]
            direction = -1
        
        xs, dist = np.linspace(self.region[0], self.region[2], grid_points,
                               retstep=True)
        
        # consider a horizontal boundary
        def func1D(x):
            return self.func((x, y0))[0]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for x0 in xs:
                try:
                    x_guess = optimize.newton(func1D, x0)
                except (RuntimeError, RuntimeWarning):
                    continue
                
                guess = np.array([x_guess, y0])
                dx, dy = self.func(guess)
                
                if norm(dx) > 1e-5 or direction*dy > 0:
                    continue
                
                if not self.point_in_region(guess):
                    continue
        
                if points.size == 0:
                    points = guess[None, :]
                elif np.all(np.abs(points - guess[None, :]).sum(axis=1) > dist):
                    points = np.vstack((points, guess))
                
        return points


    def get_steady_states_at_y_boundary(self, grid_points=32, loc='left',
                                        points=None):
        """
        finds steady state points along the y-boundary at position `loc`
        """
        
        if points is None:
            points = np.array([[]])  # array that will contain all the points
            
        if loc == 'left':
            x0 = self.region[0]
            direction = 1
        elif loc == 'right':
            x0 = self.region[2]
            direction = -1
        
        ys, dist = np.linspace(self.region[1], self.region[3], grid_points,
                               retstep=True)
        
        # consider a horizontal boundary
        def func1D(y):
            return self.func((x0, y))[1]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for y0 in ys:
                try:
                    y_guess = optimize.newton(func1D, y0)
                except (RuntimeError, RuntimeWarning):
                    continue

                guess = np.array([x0, y_guess])
                dx, dy = self.func(guess)
                
                if direction*dx > 0 or norm(dy) > 1e-5:
                    continue
                
                if not self.point_in_region(guess):
                    continue
        
                if points.size == 0:
                    points = guess[None, :]
                elif np.all(np.abs(points - guess[None, :]).sum(axis=1) > dist):
                    points = np.vstack((points, guess))
                
        return points
                    

    def get_steady_states(self, grid_points=32):
        """
        determines all steady states in the region.
        `grid_points` is the number of points to take as guesses along each
            axis.
        `region_constraint` can be a list of identifiers ('left', 'right',
            'top', 'bottom') indicating that the respective boundary poses a
            constraint on the dynamics and there may be stationary points along
            the boundary
        """
        if self.steady_states is None:

            points = np.array([[]])  # array that will contain all the points
            
            xs, dx = np.linspace(self.region[0], self.region[2], grid_points,
                                 retstep=True)
            ys, dy = np.linspace(self.region[1], self.region[3], grid_points,
                                 retstep=True)
            
            # check the border separately if requested
            if 'left' in self.region_constraint:
                points = self.get_steady_states_at_y_boundary(grid_points,
                                                              'left', points)
                xs = xs[1:]  # skip this point in future calculations
            if 'bottom' in self.region_constraint:
                points = self.get_steady_states_at_x_boundary(grid_points,
                                                              'lower', points)
                ys = ys[1:]  # skip this point in future calculations
            if 'right' in self.region_constraint:
                points = self.get_steady_states_at_y_boundary(grid_points,
                                                              'right', points)
                xs = xs[:-1]  # skip this point in future calculations
            if 'top' in self.region_constraint:
                points = self.get_steady_states_at_x_boundary(grid_points,
                                                              'upper', points)
                ys = ys[:-1]  # skip this point in future calculations
                        
            xs, ys = np.meshgrid(xs, ys)
            dist = max(dx, dy)
            
            # find all stationary points
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                for guess in zip(xs.flat, ys.flat):
                    
                    try:
                        guess = optimize.fsolve(self.func, guess, xtol=1e-8)
                    except (RuntimeError, RuntimeWarning):
                        continue
                    
                    if norm(self.func(guess)) > 1e-5:
                        continue
            
                    guess = np.array(guess)
            
                    if not self.point_in_region(guess):
                        continue
            
                    if points.size == 0:
                        points = guess[None, :]
                    elif np.all(np.abs(points - guess[None, :]).sum(axis=1) >
                                dist):
                        points = np.vstack((points, guess))
            
            # determine stability of the steady states
            stable, unstable = [], []
            for point in points:
                if self.point_is_stable(point):
                    stable.append(point)
                else:
                    unstable.append(point)
        
            stable = np.array(stable)
            unstable = np.array(unstable)
        
            self.steady_states = (stable.reshape((-1, 2)),
                                  unstable.reshape((-1, 2)))
        
        return self.steady_states
    

    def plot_steady_states(self, ax=None, color='k', **kwargs):
        """
        plots the steady states
        """
        stable, unstable = self.get_steady_states()
        
        if ax is None:
            ax = plt.gca()
        
        if stable.size > 0:
            ax.plot(
                stable[:, 0], stable[:, 1],
                'o', color=color, clip_on=False, **kwargs
            )
        if unstable.size > 0:
            ax.plot(
                unstable[:, 0], unstable[:, 1],
                'o', markeredgecolor=color, markerfacecolor='none', 
                markeredgewidth=1, clip_on=False, **kwargs
            )


    def plot_stationarity_line(self, axis=0, step=None, **kwargs):
        """
        plots the lines along which the variable plotted on `axis` is stationary
        """
        
        if step is None:
            step = self.step
        
        i_vary = 1 - axis  # index to vary
        
        # collect all start points
        points = np.concatenate(self.get_steady_states()).tolist() 
        
        # build vector for right hand side
        eps = np.zeros(2)
        eps[i_vary] = 1e-6
        
        def rhs(angle, point, step):
            """ rhs of the differential equation """
            x = point + step*np.array([np.cos(angle), np.sin(angle)])
            return self.func(x)[axis]
        
        def ensure_trajectory(point, ds):
            """ make sure we are actually on the trajectory """
            angle = np.arctan2(ds[1], ds[0])
            step = norm(ds)
            angle = optimize.newton(rhs, x0=angle, args=(point, step))
            return point + step*np.array([np.cos(angle), np.sin(angle)])
                   
        def get_traj(point, direction):
            """ retrieve an array with trajectory points """
            x0, dx = np.array(point), direction
            xs = [x0]
            
            while True:
                x0 = xs[-1]
                dx *= step/norm(dx)

                # check distances to all endpoints
                for p in points:
                    if norm(p - x0 - dx) < step:
                        xs.append(p)  # add the point to the line
                        # skip over this point in the integration
                        dx *= 2  # step over the steady state
                        points.remove(p)
                        break
              
                # make sure we're on the trajectory
                try:
                    x1 = ensure_trajectory(x0, dx)
                except (RuntimeError, RuntimeWarning):
                    break

                # check whether trajectory left the system
                if not self.point_in_region(x1, strict=True):
                    break

                xs.append(x1)
                # get step by extrapolating from last step
                dx = x1 - x0
            
            return np.array(xs)
                                
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            while len(points) > 0:
                point = points.pop()
                
                dx = solve(self.jacobian(point), eps)
    
                # follow the trajectory in both directions
                traj = np.concatenate((
                    get_traj(point, -dx)[::-1],
                    get_traj(point, +dx)[1:]
                ))
    
                if len(traj) > 3:
                    plt.plot(traj[:, 0], traj[:, 1], **kwargs)
               
        
    def plot_streamline(self, x0, ds=0.01, endpoints=None, point_margin=None,
                        ax=None, skip_initial_points=False, color='k',
                        **kwargs):
        """
        Plots a single stream line starting at x0 evolving under the flow.
        `ds` determines the step size (if it is negative we evolve back in
        time).
        """
        if ax is None:
            ax = plt.gca()

        if endpoints is None:
            endpoints = np.concatenate(self.get_steady_states())
            
        if point_margin is None:
            point_margin = 5*self.step
        
        traj = [np.array(x0)]
        
        while True:
            x = traj[-1]  # last point

            # check whether trajectory left the system
            if not self.point_in_region(x):
                break
            
            # check distances to endpoints
            if endpoints.size > 0:
                dist_to_endpoints = np.sqrt(
                    ((endpoints - x[None, :])**2).sum(axis=1)
                ).min()
                if dist_to_endpoints < point_margin:
                    break
            
            # iterate one step
            dx = np.array(self.func(x))
            traj.append(x + ds*dx/norm(dx))

        # finished iterating => plot
        if len(traj) > 1:
            traj = np.array(traj)
            if skip_initial_points:
                i_start = int(point_margin/np.abs(ds))
            else:
                i_start = 0
            plt.plot(traj[i_start::10, 0], traj[i_start::10, 1], '-',
                     color=color, **kwargs)

            # indicate direction with an arrow in the middle
            # TODO: calculate the midpoint based on actual pathlength
            i = int(0.5 * len(traj))  # midpoint
            try:
                dx = np.sign(ds)*(traj[i+5] - traj[i-5])
            except IndexError:
                dx = np.sign(ds)*(traj[i] - traj[i-1])
            dx *= 1e-2/norm(dx)
            plt.arrow(
                traj[i, 0], traj[i, 1], dx[0], dx[1],
                width=self.step/10, color=color, length_includes_head=True,
                zorder=10, clip_on=False
            )


    def plot_streamlines(self, point, angles=None, stable_direction=None,
                         **kwargs):
        """
        Plots streamlines starting from points around `point`.
        `angles` are given in degrees to avoid factors of pi
        """
        point = np.asarray(point)
        if stable_direction is None:
            stable_direction = self.point_is_stable(point)

        stable, unstable = self.get_steady_states()
        if stable_direction:
            ds = -0.01  # integration step and direction
            endpoints = unstable
        else:
            ds = 0.01  # integration step and direction
            endpoints = stable
        
        if angles is None:
            angles = np.arange(0, 360, 45)
        else:
            angles = np.asarray(angles)
        
        for angle in angles:
            # initial point: use exact forms to avoid numerical instabilities
            if angle == 0:
                dx = np.array([1, 0])
            elif angle == 90:
                dx = np.array([0, 1])
            elif angle == 180:
                dx = np.array([-1, 0])
            elif angle == 270:
                dx = np.array([0, -1])
            else:
                angle *= np.pi/180
                dx = np.array([np.cos(angle), np.sin(angle)])
                
            x0 = point + np.abs(ds)*dx
            
            self.plot_streamline(x0, ds=ds, endpoints=endpoints,
                                 skip_initial_points=True, **kwargs)

    
    def plot_heteroclinic_orbits(self, **kwargs):
        """
        Plots the heteroclinic orbits connecting different stationary states
        """
        stable, unstable = self.get_steady_states()
        points = np.concatenate((stable, unstable))
        
        # iterate through all steady states that are not border points
        for point in points:
            
            # determine stable and unstable directions
            eigenvalues, eigenvectors = eig(self.jacobian(point))
            
            # iterate through all eigenvalues
            for k, ev in enumerate(eigenvalues):
                if ev.real < 0:
                    # stable state
                    ds = -0.001
                    endpoints = unstable
                else:
                    # unstable state
                    ds = 0.001
                    endpoints = stable
                
                # start trajectories in both directions
                for dx in (ds, -ds):
                    x0 = point + 1e-6*dx*eigenvectors[:, k]
                    self.plot_streamline(x0, ds, endpoints=endpoints,
                                         skip_initial_points=True, **kwargs)



def scatter_barchart(data, labels=None, ax=None, barwidth=0.7, color=None):
    """
    Creates a plot of groups of data points, where the individual points
    are represented by a scatter plot and the mean for each group is
    depicted as a bar in a histogram style.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    if not hasattr(data[0], '__iter__'):
        data = [data]

    color = style.get_color_iter(color)
    w = 0.5*barwidth

    # make the histogram
    x = np.arange(len(data)) - w
    y = [np.mean(vals) for vals in data]
    plt.bar(
        x, y, width=barwidth, color='white', edgecolor='black', linewidth=1.
    )

    # add the scatter plot on top
    for gid, ys in enumerate(data):
        xs = np.linspace(gid-0.9*w, gid+0.9*w, len(ys))
        ax.plot(
            xs, ys, 'ko', markersize=8, markerfacecolor=next(color),
            markeredgewidth=0., markeredgecolor='white'
        )

    # ensure that the space between the bars equals the margin to the frame
    ax.set_xlim(w - 1., len(data) - w)

    # set labels
    if labels is not None:
        plt.xticks(range(len(labels)), labels)
    else:
        plt.xticks(range(len(data)))
        
        
        
def hist_logscale(data, bins=10, data_range=None, **kwargs):
    """ plots a histogram with logarithmic bin widths """
    # extract the positive part of the data
    data_pos = data[data > 0]
    
    if data_range is None:
        data_range = (data_pos.min(), data_pos.max())
    
    # try determining the bins
    try:
        bins = logspace(data_range[0], data_range[1], bins + 1)
        print(bins)
    except TypeError:
        # `bins` might have been a numpy array already
        pass
    
    res = plt.hist(data, bins=bins, **kwargs)
    plt.xscale('log')
    return res



def maskedplot(x, y, mask, *args, **kwargs):
    """
    plots a line given by points x, y using a mask
    if `mask` is NaN, no line is drawn, if the mask evaluates to True,
        a solid line is used, otherwise a dotted line is drawn.
    """
    label = kwargs.pop('label', None)
    close_gaps = kwargs.pop('close_gaps', False)
    di = 1 if close_gaps else 0

    start = 0
    res = []
    for end in range(1, len(x)):
        if mask[end] != mask[start]:
            if np.isnan(mask[start]):
                pass
            elif mask[start]:
                res.append(plt.plot(
                    x[start:end+di], y[start:end+di],
                    '-', label=label, *args, **kwargs
                ))
                label = None
            else:
                res.append(plt.plot(
                    x[start:end+di], y[start:end+di], ':', *args, **kwargs
                ))
            start = end

    # print last line
    style = '-' if mask[start] else ':'
    res.append(
        plt.plot(x[start:], y[start:], style, label=label, *args, **kwargs)
    )
    return res



def errorplot(x, y, yerr=None, fmt='', **kwargs):
    """
    Creates an error plot in which y-errors are represented by a shaded area
    instead of individual errorbars. This function accepts most arguments of the 
    traditional matplotlib `errorbar` function and translates them.
    Additionally, the following arguments are accepted
    
    `subsample` allows to plot only a fraction of the actual data points in the
        plot of the mean, while all data points are used for the envelope
        showing the errorbars 
    """
    label = kwargs.pop('label', None)
    subsample = kwargs.pop('subsample', 1)
    has_error = (yerr is not None)
    
    # plot the mean
    if fmt != 'none':
        if has_error:
            line_label = None
        else:
            line_label = label
            
        # convert data to numpy, in case they are not already. This is for
        # instance important if x or y are pandas series, in which case the
        # label might be set accidentally 
        x_arr = np.asarray(x[::subsample])
        y_arr = np.asarray(y[::subsample])
        
        # plot the line with the given properties
        line = plt.plot(x_arr, y_arr, fmt, label=line_label, **kwargs)[0]
        color = kwargs.pop('color', line.get_color())
        
    else:
        line = None
        color = kwargs.pop('color', None)
        
    # plot the deviation
    if has_error:
        alpha = kwargs.pop('alpha', 0.3)
        kwargs.pop('ls', None)  # ls only applies to centerline
        
        y = np.asarray(y)
        yerr = np.asarray(yerr)
        
        shape_err = plt.fill_between(x, y - yerr, y + yerr, color=color,
                                     edgecolors='none', alpha=alpha, 
                                     label=label, **kwargs)
    else:
        shape_err = None

    return line, shape_err



def contour_to_hatched_patches(cntrset, hatch_colors, hatch_patterns,
                               remove_contour=True):
    """ Function turning a filled contour plot into an equivalent one
    using hatches to show areas.
    Code has been taken from StackOverflow!
    """
    from itertools import cycle
    from matplotlib.patches import PathPatch

    ax = plt.gca()
    patches_list = []
    for pathcollection in cntrset.collections:
        patches_list.append([PathPatch(p) for p in pathcollection.get_paths()])
        if remove_contour:
            pathcollection.remove()

    hatch_colors = cycle(hatch_colors)
    hatch_patterns = cycle(hatch_patterns)

    for patches, _, hp in zip(patches_list, hatch_colors, hatch_patterns):
        for p in patches:
            p.set_fc("none")
            p.set_ec("k")
            p.set_hatch(hp)
            ax.add_patch(p)



def get_hatched_image(values, stripe_width=0.05, orientation='/'):
    """
    Takes three dimensional boolean data and projects out the last dimension,
    by using hatched regions to symbolize parts, where several variables are
    true
    """

    if stripe_width < 1:
        stripe_width = int(stripe_width*max(values.shape[:2]))

    # build orientation list
    dimensions = len(values[0, 0, :])
    if hasattr(orientation, '__iter__'):
        orientations = itertools.cycle(orientation)
        orientations = list(itertools.islice(orientations, dimensions))
    else:
        orientations = [orientation]*dimensions

    # convert the values and calculate the number of values
    values = np.atleast_3d(values)
    nums = values.sum(2).astype(int)

    res = np.zeros(values.shape[:2])
    for x, y in np.ndindex(*res.shape):
        if nums[x, y] == 0:
            res[x, y] = -1
        else:
            orientation = orientations[nums[x, y] - 1]
            # choose a color index based on the current stripe
            if orientation == '\\':
                i = ((x - y) % stripe_width)*nums[x, y]//stripe_width
            elif orientation == '-':
                i = (x % stripe_width)*nums[x, y]//stripe_width
            elif orientation == '/':
                i = ((x + y) % stripe_width)*nums[x, y]//stripe_width
            elif orientation == '|':
                i = (y % stripe_width)*nums[x, y]//stripe_width
            else:
                raise ValueError(
                    'Allowed stripe orientation values: /, -, \\,  |'
                )
            # choose the color from the current values
            res[x, y] = np.nonzero(values[x, y, :])[0][i]

    return res



if __name__ == "__main__":
    print('This file is intended to be used as a module.')
    print('This code serves as a test for the defined methods.')

    tests = (
        'contour_plot',
        'scatter_barchart',
        'hist_logscale',
        'hatched_image'
    )


    if 'contour_plot' in tests:
        xs, ys = np.meshgrid(np.linspace(0, 10, 201), np.linspace(0, 10, 201))
        z = np.empty((201, 201, 2))
        z[:, :, 0] = np.sin(xs) > 0.5*np.cos(ys)
        z[:, :, 1] = np.cos(ys) > 0.2*np.sin(xs)
    
        z[range(100), range(100), 0] = np.nan
    
        make_contour_plot_hatched(
            z, ('r', 'g'),
            stripe_width=0.04, stripe_orientation=('\\', '/'),
            interpolation='none',
            boundary_style={'linewidths': 3}
        )
        plt.show()


    if 'scatter_barchart' in tests:
        # testplot with scatter_barchart
        plt.clf()
        test_data = np.zeros((5, 10))
        for i in range(5):
            test_data[i, :] = i + np.random.rand(10)

        scatter_barchart(test_data, labels=('a', 'b', 'c', 'd', 'e'))

        plt.show()

    if 'hist_logscale' in tests:
        test_data = np.exp(np.random.uniform(0, 10, 1000))
        hist_logscale(test_data)
        plt.show()

    if 'hatched_image' in tests:
        test_x, test_y = np.meshgrid(np.linspace(0, 10, 201),
                                     np.linspace(0, 10, 201))
        z = np.empty((201, 201, 2), np.bool)
        z[:, :, 0] = np.sin(test_x) > 0
        z[:, :, 1] = np.cos(test_y) > 0

        img = get_hatched_image(z, stripe_width=12, orientation='\\')
        plt.imshow(img)
        plt.show()
        