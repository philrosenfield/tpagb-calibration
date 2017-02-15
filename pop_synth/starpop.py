from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from .. import utils
from .. import fileio

from ..angst_tables import angst_data
from ..utils.plotting_utils import (scatter_contour, colorify, make_hess,
    plot_hess, plot_cmd_redding_vector)
from ..trilegal import get_stage_label
from ..utils.astronomy_utils import get_dmodAv, mag2Mag, Mag2mag

__all__ = ['StarPop', 'plot_cmd', 'color_by_arg', 'stars_in_region']


class StarPop(object):
    def __init__(self):
        self.data = None

    def plot_cmd(self, *args, **kwargs):
        '''
        plot a stellar color-magnitude diagram
        see :func: plot_cmd
        '''
        return plot_cmd(self, *args, **kwargs)

    def color_by_arg(self, *args, **kwargs):
        """
        see :func: color_by_arg
        """
        return color_by_arg(self, *args, **kwargs)

    def redding_vector(self, dmag=1., **kwargs):
        """ Add an arrow to show the reddening vector """

        return plot_cmd_redding_vector(self.filter1, self.filter2,
                                       self.photsys, dmag=dmag, **kwargs)

    def scatter_hist(self, *args, **kwargs):
        """
        see :func: scatter_hist
        """
        return scatter_hist(self, *args, **kwargs)

    def decorate_cmd(self, mag1_err=None, mag2_err=None, trgb=False, ax=None,
                     reddening=True, dmag=0.5, text_kw={}, errors=True,
                     cmd_errors_kw={}, filter1=None, text=True):
        """ add annotations on the cmd, such as reddening vector, typical errors etc
        """

        self.redding_vector(dmag=dmag, ax=ax)

        if errors is True:
            cmd_errors_kw['ax'] = ax
            self.cmd_errors(**cmd_errors_kw)

        self.text_on_cmd(ax=ax, **text_kw)

        if trgb is True:
            if filter1 is None:
                self.put_a_line_on_it(ax, self.trgb)
            else:
                self.put_a_line_on_it(ax, self.trgb, filter1=filter1, consty=False)

    def put_a_line_on_it(self, ax, val, consty=True, color='black',
                         ls='--', lw=2, annotate=True, filter1=None,
                         annotate_fmt='$TRGB=%.2f$', **kwargs):
        """
        if consty is True: plots a constant y value across ax.xlims().
        if consty is False: plots a constant x on a plot of y vs x-y
        """
        (xmin, xmax) = ax.get_xlim()
        (ymin, ymax) = ax.get_ylim()
        xarr = np.linspace(xmin, xmax, 20)
        # y axis is magnitude...
        yarr = np.linspace(ymin, ymax, 20)
        if consty is True:
            # just a contsant y value over the plot range of x.
            ax.hlines(val, xmin, xmax, color=color, lw=lw)
            new_xarr = xarr
        if consty is False:
            # a plot of y vs x-y and we want to mark
            # where a constant value of x is
            # e.g, f814w vs f555-f814; val is f555
            new_xarr = val - yarr
            # e.g, f555w vs f555-f814; val is f814
            if filter1 is not None:
                yarr = xarr + val
                new_xarr = xarr
            ax.plot(new_xarr, yarr, ls, color=color, lw=lw)

        if annotate is True:
            xy = (new_xarr[-1] - 0.1, yarr[-1] - 0.2)
            ax.annotate(annotate_fmt % val, xy=xy, ha='right', fontsize=16, **kwargs)
        return new_xarr, yarr

    def text_on_cmd(self, extra=None, ax=None, distance_av=True, **kwargs):
        """ add a text on the cmd
        """
        ax = ax or plt.gca()

        if distance_av is True:
            strings = '$%s$ $\mu=%.3f$ $A_v=%.2f$' % (self.target.upper(), self.dmod, self.Av)
            offset = .17
        else:
            strings = '$%s$' % self.target.upper().replace('-DEEP', '').replace('-', '\!-\!')
            offset = .09
        if extra is not None:
            strings += ' %s' % extra
            offset = 0.2
        for string in strings.split():
            offset -= 0.04
            ax.text(0.95, offset, string, transform=ax.transAxes, ha='right',
                    fontsize=16, color='black', **kwargs)

    def annotate_cmd(self, yval, string, offset=0.1, text_kw={}, ax=None):
        ax = ax or plt.gca()
        ax.text(ax.get_xlim()[0] + offset, yval - offset, string, **text_kw)

    def make_hess(self, binsize, absmag=False, useasts=False, slice_inds=None,
                  **kwargs):
        '''
        adds a hess diagram of color, mag2 or Color, Mag2 (if absmag is True).
        if useasts is true will use ast_mags.

        slice_inds will slice the arrays and only bin those stars, there
        is no evidence besides the hess tuple itself that hess is not of the
        full cmd.

        See `:func: helpers.make_hess`
        '''
        if absmag is True:
            col = self.Color
            mag = self.Mag2
        elif useasts is True:
            col = self.ast_color[self.rec]
            mag = self.ast_mag2[self.rec]
        else:
            col = self.color
            mag = self.mag2
        if slice_inds is not None:
            col = col[slice_inds]
            mag = mag[slice_inds]
        self.hess = make_hess(col, mag, binsize, **kwargs)
        return

    def hess_plot(self, fig=None, ax=None, colorbar=False, **kwargs):
        '''
        Plots a hess diagram with imshow.

        See `:func: helpers.plot_hess`
        '''
        if not hasattr(self, 'hess'):
            raise AttributeError('run self.make_hess before plotting')

        if hasattr(self, 'filter2') and hasattr(self, 'filter1'):
            filter1 = self.filter1
            filter2 = self.filter2
        else:
            filter1 = None
            filter2 = None

        ax = plot_hess(self.hess, ax=ax, filter1=filter1, filter2=filter2,
                       colorbar=colorbar, **kwargs)
        return ax

    def get_header(self):
        '''
        utility for writing data files, sets header attribute and returns
        header string.
        '''
        try:
            names = list(utils.sort_dict(self.key_dict).keys())
        except AttributeError:
            names = self.data.dtype.names
        self.header = '# %s' % ' '.join(names)
        return self.header

    def delete_data(self, data_names=None):
        '''
        for wrapper functions, I don't want gigs of data stored when they
        are no longer needed.
        '''
        if data_names is None:
            data_names = ['data', 'mag1', 'mag2', 'color', 'stage', 'ast_mag1',
                          'ast_mag2', 'ast_color', 'rec']
        for data_name in data_names:
            if hasattr(self, data_name):
                self.__delattr__(data_name)
            if hasattr(self, data_name.title()):
                self.__delattr__(data_name.title())

    def histoattr(self, attr, bins=10, inds=None):
        '''
        call np.histogram for a given attribute in self.data sliced by inds
        '''
        if inds is None:
            inds = np.arange(self.data.size)

        hist, bins = np.histogram(self.data[attr][inds], bins=bins)

        return hist, bins

    def convert_mag(self, dmod=0., Av=0., target=None, shift_distance=False,
                    useasts=False):
        '''
        convert from mag to Mag or from Mag to mag or just shift distance.
        pass dmod, Av, or use AngstTables to look it up from target.
        shift_distance: for the possibility of doing dmod, Av fitting of model
        to data the key here is that we re-read the mag from the original data
        array.

        useasts only work with shift_distance is true.
        It will calculate the original dmod and av from self, and then shift
        that to the new dmod av. there may be a faster way, but this is just
        multiplicative effects.
        Without shift_distance: Just for common usage. If trilegal was given a
        dmod, it will swap it back to Mag, if it was done at dmod=10., will
        shift to given dmod. mag or Mag attributes are set in __init__.

        '''
        check = [(dmod + Av == 0.), (target is None)]

        #assert False in check, 'either supply dmod and Av or target'

        if check[0] is True:
            filters = ','.join((self.filter1, self.filter2))
            if target is not None:
                print('converting mags with angst table using %s' % target)
                self.target = target
            elif hasattr(self, 'target'):
                print('converting mags with angst table using initialized %s' % self.target)

            tad = angst_data.get_tab5_trgb_av_dmod(self.target, filters)
            __, self.Av, self.dmod = tad

        else:
            self.dmod = dmod
            self.Av = Av

        mag_covert_kw = {'Av': self.Av, 'dmod': self.dmod}

        if shift_distance is True:
            if useasts is True:
                am1 = self.ast_mag1
                am2 = self.ast_mag2
                old_dmod, old_Av = get_dmodAv(self)
                old_mag_covert_kw = {'Av': old_Av, 'dmod': old_dmod}
                M1 = mag2Mag(am1, self.filter1, self.photsys, **old_mag_covert_kw)
                M2 = mag2Mag(am2, self.filter2, self.photsys, **old_mag_covert_kw)
            else:
                M1 = self.data.get_col(self.filter1)
                M2 = self.data.get_col(self.filter2)
            self.mag1 = Mag2mag(M1, self.filter1, self.photsys, **mag_covert_kw)
            self.mag2 = Mag2mag(M2, self.filter2, self.photsys, **mag_covert_kw)
            self.color = self.mag1 - self.mag2
        else:
            if hasattr(self, 'mag1'):
                self.Mag1 = mag2Mag(self.mag1, self.filter1, self.photsys, **mag_covert_kw)
                self.Mag2 = mag2Mag(self.mag2, self.filter2, self.photsys, **mag_covert_kw)
                self.Color = self.Mag1 - self.Mag2

                if hasattr(self, 'trgb'):
                    self.Trgb = mag2Mag(self.trgb, self.filter2, self.photsys, **mag_covert_kw)

            if hasattr(self, 'Mag1'):
                self.mag1 = Mag2mag(self.Mag1, self.filter1, self.photsys, **mag_covert_kw)
                self.mag2 = Mag2mag(self.Mag2, self.filter2, self.photsys, **mag_covert_kw)
                self.color = self.mag1 - self.mag2

    def add_data(self, names, data):
        '''
        add columns to self.data, update self.key_dict
        see numpy.lib.recfunctions.append_fields.__doc__

        Parameters
        ----------
        names : string, sequence
            String or sequence of strings corresponding to the names
            of the new fields.
        data : array or sequence of arrays
            Array or sequence of arrays storing the fields to add to the base.

        Returns
        -------
        header
        '''

        self.data = utils.add_data(self.data, names, data)

        # update key_dict
        header = self.get_header()
        header += ' ' + ' '.join(names)
        col_keys = header.replace('#', '').split()
        self.key_dict = dict(zip(col_keys, range(len(col_keys))))
        return header

    def slice_data(self, keys, inds):
        '''slice already set attributes by some index list.'''
        for d in keys:
            if hasattr(self, d):
                self.__setattr__(d, self.__dict__[d][inds])
            if hasattr(self, d.title()):
                d = d.title()
                self.__setattr__(d, self.__dict__[d][inds])

    def double_gaussian_contamination(self, all_verts, dcol=0.05, Color=None,
                                      Mag2=None, color_sep=None, diag_plot=False,
                                      absmag=False, thresh=5):
        '''
        This function fits a double gaussian to a color histogram of stars
        within the <maglimits> and <colorlimits> (tuples).

        It then finds the intersection of the two gaussians, and the fraction
        of each integrated gaussian that crosses over the intersection color
        line.
        '''

        # the indices of the stars within the MS/BHeB regions

        # poisson noise to compare with contamination
        if Color is None:
            if absmag is True:
                Color = self.Color
                Mag2 = self.Mag2
            else:
                Color = self.color
                Mag2 = self.mag2

        points = np.column_stack((Color, Mag2))
        all_inds, = np.nonzero(utils.points_inside_poly(points, all_verts))

        if len(all_inds) <= thresh:
            print('not enough points found within verts')
            return np.nan, np.nan, np.nan, np.nan, np.nan

        poission_noise = np.sqrt(float(len(all_inds)))

        # make a color histogram
        #dcol = 0.05
        color = Color[all_inds]
        col_bins = np.arange(color.min(), color.max() + dcol, dcol)
        #nbins = np.max([len(col_bins), int(poission_noise)])
        hist = np.histogram(color, bins=col_bins)[0]

        # uniform errors
        err = np.zeros(len(col_bins[:1])) + 1.

        # set up inputs
        hist_in = {'x': col_bins[1:], 'y': hist, 'err': err}

        # set up initial parameters:
        # norm = max(hist),
        # mean set to be half mean, and 3/2 mean,
        # sigma set to be same as dcol spacing...
        p0 = [np.nanmax(hist) / 2., np.mean(col_bins[1:]) - np.mean(col_bins[1:]) / 2., dcol,
              np.nanmax(hist) / 2., np.mean(col_bins[1:]) + np.mean(col_bins[1:]) / 2., dcol]

        mp_dg = utils.mpfit(utils.mp_double_gauss, p0, functkw=hist_in, quiet=True)
        if mp_dg.covar is None:
            print('not double gaussian')
            return 0., 0., poission_noise, float(len(all_inds)), color_sep
        else:
            perc_err = (np.array(mp_dg.perror) - np.array(mp_dg.params)) / np.array(mp_dg.params)
            if np.sum([p ** 2 for p in perc_err]) > 10.:
                print('not double guassian, errors too large')
                return 0., 0., poission_noise, float(len(all_inds)), color_sep
        # take fit params and apply to guassians on an arb color scale
        color_array = np.linspace(col_bins[0], col_bins[-1], 1000)
        g_p1 = mp_dg.params[0: 3]
        g_p2 = mp_dg.params[3:]
        gauss1 = utils.gaussian(color_array, g_p1)
        gauss2 = utils.gaussian(color_array, g_p2)
        print(g_p1[1], g_p2[1])
        # color separatrion is the intersection of the two gaussians..
        #double_gauss = gauss1 + gauss2
        #between_peaks = np.arange(
        min_locs = utils.find_peaks(gauss1 + gauss2)['minima_locations']
        g1, g2 = np.sort([g_p1[1], g_p2[2]])
        ginds, = np.nonzero( (color_array > g1) & (color_array < g2))
        #ginds2, = np.nonzero(gauss2)
        #ginds = list(set(ginds1) & set(ginds2))
        min_locs = np.argmin(np.abs(gauss1[ginds] - gauss2[ginds]))
        print(min_locs)
        auto_color_sep = color_array[ginds][min_locs]
        print(auto_color_sep)
        if auto_color_sep == 0:
            auto_color_sep = np.mean(col_bins[1:])
            print('using mean as color_sep')
        if color_sep is None:
            color_sep = auto_color_sep
        else:
            print('you want color_sep to be %.4f, I found it at %.4f' % (color_sep, auto_color_sep))

        # find contamination past the color sep...
        g12_Integral = integrate.quad(utils.double_gaussian, -np.inf, np.inf, mp_dg.params)[0]

        try:
            norm = float(len(all_inds)) / g12_Integral
        except ZeroDivisionError:
            norm = 0.

        g1_Integral = integrate.quad(utils.gaussian, -np.inf, np.inf, g_p1)[0]
        g2_Integral = integrate.quad(utils.gaussian, -np.inf, np.inf, g_p2)[0]

        g1_Int_colsep = integrate.quad(utils.gaussian, -np.inf, color_sep, g_p1)[0]
        g2_Int_colsep = integrate.quad(utils.gaussian, color_sep, np.inf, g_p2)[0]

        left_in_right = (g1_Integral - g1_Int_colsep) * norm
        right_in_left = (g2_Integral - g2_Int_colsep) * norm
        # diagnostic
        #print color_sep
        if diag_plot is True:
            fig1, ax1 = plt.subplots()
            ax1.plot(col_bins[1:], hist, ls='steps', lw=2)
            ax1.plot(col_bins[1:], hist, 'o')
            ax1.plot(color_array, utils.double_gaussian(color_array, mp_dg.params))
            ax1.plot(color_array, gauss1)
            ax1.plot(color_array, gauss2)
            #ax1.set_ylim((0, 100))
            ax1.set_xlim(color.min(), color.max())
            ax1.set_xlabel('$%s-%s$' % (self.filter1, self.filter2), fontsize=20)
            ax1.set_ylabel('$\#$', fontsize=20)
            ax1.set_title('%s Mean Mag2: %.2f, Nbins: %i' % (self.target,
                                                             np.mean(np.array(all_verts)[:, 1]),
                                                             len(col_bins)))
            ax1.vlines(color_sep, *ax1.get_ylim())
            ax1.text(0.1, 0.95, 'left in right: %i' % left_in_right, transform=ax1.transAxes)
            ax1.text(0.1, 0.90, 'right in left: %i' % right_in_left, transform=ax1.transAxes)

            fig_fname = 'heb_contamination_%s_%s_%s_mag2_%.2f.png' % (self.filter1, self.filter2, self.target, np.mean(np.array(all_verts)[:, 1]))

            fig1.savefig(fig_fname)
            print('wrote {0:s}'.format(fig_fname))

        return left_in_right, right_in_left, poission_noise, float(len(all_inds)), color_sep

    def stars_in_region(self, *args, **kwargs):
        '''
        counts stars in a region.
        see :func: stars_in_region
        '''
        return stars_in_region(*args, **kwargs)

    def write_data(self, outfile, overwrite=False, hdf5=False, slice_inds=None):
        '''call fileio.savetxt to write self.data'''
        data = self.data
        from astropy.table import Table
        if slice_inds is not None:
            data = self.data[slice_inds]
        if not hdf5:
            if outfile.endswith('.fits'):
                tbl = Table(data=data)
                tbl.write(outfile, overwrite=overwrite)
            else:
                if overwrite or not os.isfile(outfile):
                    np.savetxt(outfile, data, fmt='%5g', header=self.get_header())
        else:
            if not outfile.endswith('.hdf5'):
                outfile = fileio.replace_ext(outfile, '.hdf5')
            tbl = Table(data)
            tbl.write(outfile, format='hdf5', path='data', compression=True,
                      overwrite=overwrite)
            print('wrote {0:s}'.format(outfile))

        return


def stars_in_region(ymag, mag_dim, mag_bright, mag1=None, verts=None,
                    col_min=None, col_max=None, color=None):
    '''
    counts stars in a region of a CMD or LF

    Parameters
    ----------
    mag1, mag2 : array mag1 optional
        arrays of star mags. If mag1 is supplied stars are assumed to be in a
        CMD, not LF.

    mag_dim, mag_bright : float, float
        faint and bright limits of mag2.

    col_min, col_max : float, float optional
        color min and max of CMD box.

    verts : array
        array shape 2, of verticies of a CMD polygon to search within

    Returns
    -------
    inds : array
        indices of mag2 inside LF or
        indices of mag1 and mag2 inside CMD box or verts
    '''
    if mag_dim < mag_bright:
        mag_dim, mag_bright = mag_bright, mag_dim

    if verts is None:
        if col_min is None:
            return utils.between(ymag, mag_dim, mag_bright)
        else:
            verts = np.array([[col_min, mag_dim],
                              [col_min, mag_bright],
                              [col_max, mag_bright],
                              [col_max, mag_dim],
                              [col_min, mag_dim]])

    if color is None:
        color = mag1 - ymag
    points = np.column_stack((color, ymag))
    inds, = np.nonzero(utils.points_inside_poly(points, verts))

    return inds


def plot_cmd(starpop, color, mag, ax=None, xlim=None, ylim=None, xlabel=None,
             ylabel=None, contour_kwargs={}, scatter_kwargs={}, plot_kwargs={},
             scatter=True, levels=5, threshold=75, contour_lw={},
             color_by_arg_kw={}, slice_inds=None, hist_bin_res=0.05,
             log_counts=False):
    '''
    plot a stellar color-magnitude diagram

    Parameters
    ----------
    starpop: StarPop instance
        population to plot

    color: ndarray, dtype=float, ndim=1
        colors of the stars

    mag: ndarray, dtype=float, ndim=1
        magnitudes of the stars

    ax: plt.Axes instance, optional (default=plt.gca())
        axes in which it will make the plot

    xlim: optional
        if set, adjust the limits on the color axis

    ylim: optional
        if set, adjust the limits on the magnitude axis

    xlabel: str, optional
        label of the color axis

    ylabel: str, optional
        label of the magntiude axis

    contour_args: dict, optional
        keywords for contour plot

    scatter_args: dict, optional
        keywords for scatter

    plot_args: dict, optional
        kweywords for plot

    scatter: bool, optional (default=True)
        CMD will be generated with plt.scatter

    levels: optional (default=5)

    threshold: optional (default=75)

    contour_lw: optional

    color_by_arg_kw: optional

    slice_inds: slice instance, optional
        if set, cull the data and only consider this slice

    hist_bin_res: optional (default=0.05),

    log_counts: bool, optional (default=False)
    '''
    ax = ax or plt.gca()

    if slice_inds is not None:
        color = color[slice_inds]
        mag = mag[slice_inds]

    if xlim is None:
        xlim = (color.min(), color.max())

    if ylim is None:
        ylim = (mag.max(), mag.min())

    if len(color_by_arg_kw) != 0:
        scatter = False
        color_by_arg(starpop, ax=ax, **color_by_arg_kw)

    if scatter is True:
        _contour_kwargs = {'cmap': plt.cm.gray_r, 'zorder': 100}
        _contour_kwargs.update(**contour_kwargs)

        _scatter_kwargs = {'marker': '.', 'color': 'black', 'alpha': 0.2,
                           'edgecolors': 'none', 'zorder': 1}
        _scatter_kwargs.update(**scatter_kwargs)

        if type(hist_bin_res) is list:
            hist_bin_res_c, hist_bin_res_m = hist_bin_res
        else:
            hist_bin_res_c = hist_bin_res
            hist_bin_res_m = hist_bin_res

        ncolbin = int(np.diff((np.nanmin(color), np.nanmax(color))) / hist_bin_res_c)
        nmagbin = int(np.diff((np.nanmin(mag), np.nanmax(mag))) / hist_bin_res_m)
        #print(ncolbin, nmagbin)

        scatter_contour(color, mag, bins=[ncolbin, nmagbin], levels=levels,
                        threshold=threshold, log_counts=log_counts,
                        plot_args=_scatter_kwargs,
                        contour_args=_contour_kwargs, ax=ax)
    else:
        #simple plotting
        _plot_kwargs = {'marker': '.', 'color': 'black', 'mew': 0., 'lw': 0, 'rasterize': True}
        _plot_kwargs.update(**plot_kwargs)
        ax.plot(color, mag, **_plot_kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if plt.isinteractive():
        plt.draw()

    return ax

    def make_phot(self, fname='phot.dat'):
        '''
        makes phot.dat input file for match, a list of V and I mags.
        '''
        np.savetxt(fname, np.column_stack((self.mag1, self.mag2)), fmt='%.4f')


def scatter_hist(starpop, xdata, ydata, coldata='stage', xbins=50, ybins=50,
                 slice_inds=None, xlim=None, ylim=None, clim=None,
                 discrete=False, scatter_kw={}):
    """ """
    import palettable
    import matplotlib.gridspec as gridspec

    def side_hist(ax, data, cdata, bins=50, cols=None, binsx=True,
                  discrete=False, cbins=10):
        """
        Add a histogram of data to ax made from inds of unique values of cdata

        """
        if discrete:
            if type(cbins) == int:
                hist, cbins = np.histogram(cdata, bins=cbins)
            dinds = np.digitize(cdata, cbins)
            uinds = range(len(cbins))
        else:
            uinds = np.unique(cdata)
            dinds = np.digitize(cdata, uinds)

        cols = palettable.get_map('Spectral', 'Diverging',
                                  len(uinds)).mpl_colors

        for j, i in enumerate(uinds):
            hist, hbins = np.histogram(data[dinds == i], bins=bins)
            if binsx:
                x = hbins[1:]
                y = hist
            else:
                y = hbins[1:]
                x = hist
            ax.plot(x, y, linestyle='steps-pre', color=cols[j], zorder=100-j)
        return ax

    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(3, 3)
    axt = plt.subplot(gs[0,:-1])
    axr = plt.subplot(gs[1:,-1])
    ax = plt.subplot(gs[1:,:-1])


    if type(xdata) is str:
        ax.set_xlabel(xdata.replace('_', '\_'))
        xdata = starpop.data[xdata]

    if type(ydata) is str:
        ax.set_ylabel(ydata.replace('_', '\_'))
        ydata = starpop.data[ydata]

    collabel = None
    if type(coldata) is str:
        collabel = coldata.replace('_', '\_')
        coldata = starpop.data[coldata]

    if slice_inds is not None:
        xdata = xdata[slice_inds]
        ydata = ydata[slice_inds]
        coldata = coldata[slice_inds]

    scatter_kw = dict({'edgecolors': 'none', 'cmap': plt.cm.Spectral}.items() \
                       + scatter_kw.items())
    l = ax.scatter(xdata, ydata, c=coldata,  **scatter_kw)

    axt = side_hist(axt, xdata, coldata, bins=xbins, discrete=discrete)
    axr = side_hist(axr, ydata, coldata, bins=ybins, binsx=False,
                    discrete=discrete)

    axt.set_yscale('log')
    axr.set_xscale('log')
    if xlim is not None:
        axt.set_xlim(xlim)
    ax.set_xlim(axt.get_xlim())

    if ylim is not None:
        axr.set_ylim(ylim)
    ax.set_ylim(axr.get_ylim())

    if clim is not None:
        l.set_clim(clim)

    axt.set_ylabel('$\#$')
    axr.set_xlabel('$\#$')

    axt.tick_params(labelbottom=False, labeltop=True, right=False)
    axr.tick_params(labelright=True, labelleft=False, top=False)
    gs.update(left=0.1, right=0.9, wspace=0, hspace=0)

    if collabel == 'stage':
        axtr = plt.subplot(gs[0, -1])
        cols = palettable.get_map('Spectral', 'Diverging',
                                  9).mpl_colors
        [axtr.plot(0, 0, label=get_stage_label()[i], color=cols[i])
         for i in range(len(cols))]
        axtr.tick_params(labelbottom=False, labelleft=False)
        axtr.grid()  # should turn it off!!
        plt.legend(mode='expand', ncol=2, frameon=False)
        axs = [axt, ax, axr, axtr]
    else:
        axs = [axt, ax, axr]


    return axs


def color_by_arg(starpop, xdata, ydata, coldata, bins=None, cmap=None, ax=None,
                 fig=None, labelfmt='$%.3f$', xlim=None, ylim=None, clim=None,
                 slice_inds=None, legend=True, discrete=False, skw={}):
    """
    Parameters
    ----------
    starpop : StarPop instance
        population to plot

    xdata, ydata, coldata : array or str
        xdata array or column name of starpop.data
        if column name, will add xlabel, ylabel, or colorbar label.

    cmap : mpl colormap instance
        colormap to use

    ax : plt.Axes instance

    bins : int or array
        if discrete, passed to np.histogram bins=bins

    labelfmt : str, optional (default='$%.3f$')
        colorbar label format

    xlim, ylim : tuples
        if set, adjust the limits on the x and y axes

    slice_inds : list or array
        slice the data

    Returns
    -------
    ax : plt.Axes instance
    """
    def latexify(string):
        return r'${}$'.format(string.replace('_', '\_'))

    ax = ax or plt.gca()

    if type(xdata) is str:
        ax.set_xlabel(latexify(xdata))
        xdata = starpop.data[xdata]

    if type(ydata) is str:
        ax.set_ylabel(latexify(ydata))
        ydata = starpop.data[ydata]

    collabel = None
    if type(coldata) is str:
        collabel = latexify(coldata)
        coldata = starpop.data[coldata]

    if slice_inds is not None:
        xdata = xdata[slice_inds]
        ydata = ydata[slice_inds]
        coldata = coldata[slice_inds]

    if discrete:
        if bins is None:
            bins = 10
        # need the bins to be an array to use digitize.
        if type(bins) == int:
            hist, bins = np.histogram(coldata, bins=bins)
        inds = np.digitize(coldata, bins)
        colors, scalarmap = colorify(inds, cmap=cmap)
        lbls = [ labelfmt % bk for bk in bins ]  # bins are left bin edges.

        # fake out the legend...
        if labelfmt not in ['', 'None', None]:
            for color, label in zip(colors, lbls):
                ax.plot([999], [999], 'o', color=color, mec=color, label=label,
                        visible=False)
        ax.scatter(xdata[inds], ydata[inds], marker='o', s=15, edgecolors='none',
                   color=colors[inds], alpha=0.3)

        ax.legend(loc=0, numpoints=1, frameon=False)

    else:

        if bins is not None:
            if cmap is None:
                cmap = plt.get_cmap('Spectral', bins)
            cmap.set_under('gray')
            cmap.set_over('gray')
        else:
            cmap = cmap or plt.cm.Paired

        l = ax.scatter(xdata, ydata, c=coldata, marker='o', s=15,
                       edgecolors='none', cmap=cmap, **skw)

        c = plt.colorbar(l, ax=ax)
        if collabel is not None:
            c.set_label(collabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if clim is not None:
        l.set_clim(clim)
    return ax
