""" All about Artificial star tests """
from __future__ import print_function
import argparse
import logging
import os
from astropy.io import fits
import sys

import matplotlib.pylab as plt
import numpy as np

from scipy.interpolate import interp1d

from ..utils import astronomy_utils, parse_pipeline

logger = logging.getLogger(__name__)

__all__ = ['ast_correct_starpop', 'ASTs']

plt.style.use('ggplot')


def ast_correct_starpop(sgal, fake_file=None, outfile=None, overwrite=False,
                        asts_obj=None, correct_kw={}, diag_plot=False,
                        plt_kw={}, hdf5=True, correct='both',
                        filterset=0):
    '''
    correct mags with artificial star tests, finds filters by fake_file name

    Parameters
    ----------
    sgal : galaxies.SimGalaxy or StarPop instance
        must have apparent mags (corrected for dmod and Av)

    fake_file : string
         matchfake file

    outfile : string
        if sgal, a place to write the table with ast_corrections

    overwrite : bool
        if sgal and outfile, overwite if outfile exists

    asts_obj : AST instance
        if not loading from fake_file

    correct_kw : dict
        passed to ASTs.correct important to consider, dxy, xrange, yrange
        see AST.correct.__doc__

    diag_plot : bool
        make a mag vs mag diff plot

    plt_kw :
        kwargs to pass to pylab.plot

    correct : 'both' 'filter1' 'filter2'
        sepcifiy which filters get corrections

    Returns
    -------
    adds corrected mag1 and mag2

    If sgal, adds columns to sgal.data
    '''
    fmt = '{}_cor'
    if correct == 'all':
        correct = 'filter1 filter2 filter3 filter4'
    if correct == 'both':
        if filterset == 0:
            correct = 'filter1 filter2'
        else:
            correct = 'filter3 filter4'

    if asts_obj is None:
        sgal.fake_file = fake_file
        _, filter1, filter2 = parse_pipeline(fake_file)
        if fmt.format(filter1) in sgal.data.keys() or fmt.format(filter2) in sgal.data.keys():
            errfmt = '{}, {} ast corrections already in file.'
            logger.warning(errfmt.format(filter1, filter2))
            return sgal.data[fmt.format(filter1)], sgal.data[fmt.format(filter2)]
        ast = ASTs(fake_file, filters=[filter1, filter2], usecols=usecols)
    else:
        ast = asts_obj

    default = {'dxy': (0.2, 0.15)}
    default.update(correct_kw)
    correct_kw = default
    if fmt.format(ast.filter1) in sgal.data.dtype.names:
        correct = correct.replace('filter1', '')
    if fmt.format(ast.filter2) in sgal.data.dtype.names:
        correct = correct.replace('filter2', '')

    if '3' in correct:
        if fmt.format(ast.filter3) in sgal.data.dtype.names:
            correct = correct.replace('filter3', '')
    if '4' in correct:
        if fmt.format(ast.filter4) in sgal.data.dtype.names:
            correct = correct.replace('filter4', '')

    names = []
    data = []
    if 'filter1' in correct or 'filter2' in correct:
        cor_mag1, cor_mag2 = ast.correct(sgal.data[ast.filter1],
                                         sgal.data[ast.filter2], **correct_kw)
        if 'filter1' in correct:
            names.append(fmt.format(ast.filter1))
            data.append(cor_mag1)
        if 'filter2' in correct:
            names.append(fmt.format(ast.filter2))
            data.append(cor_mag2)
    if 'filter3' in correct or 'filter4' in correct:
        correct_kw['filterset'] = 1
        cor_mag3, cor_mag4 = ast.correct(sgal.data[ast.filter3],
                                         sgal.data[ast.filter4], **correct_kw)
        if 'filter3' in correct:
            names.append(fmt.format(ast.filter3))
            data.append(cor_mag3)
        if 'filter4' in correct:
            names.append(fmt.format(ast.filter4))
            data.append(cor_mag4)

    if len(data) == 0:
        logger.info('no corrections to add')
        return
    logger.info('adding corrections for {}'.format(correct))

    sgal.add_data(names, data)

    if outfile is not None:
        sgal.write_data(outfile, overwrite=overwrite, hdf5=hdf5)

    if diag_plot:
        from ..fileio.fileIO import replace_ext
        plt_kw = dict({'color': 'navy', 'alpha': 0.3, 'label': 'sim'}.items() \
                      + plt_kw.items())
        axs = ast.magdiff_plot()
        mag1diff = cor_mag1 - mag1
        mag2diff = cor_mag2 - mag2
        rec, = np.nonzero((np.abs(mag1diff) < 10) & (np.abs(mag2diff) < 10))
        axs[0].plot(mag1[rec], mag1diff[rec], '.', **plt_kw)
        axs[1].plot(mag2[rec], mag2diff[rec], '.', **plt_kw)
        if 'label' in plt_kw.keys():
            [ax.legend(loc=0, frameon=False) for ax in axs]
        plt.savefig(replace_ext(outfile, '_ast_correction.png'))
    return data


class ASTs(object):
    '''class for reading and using artificial stars'''
    def __init__(self, filename, filters=None, filt_extra=''):
        '''
        if filename has 'match' in it will assume this is a matchfake file.
        if filename has .fits extention will assume it's a binary fits table.
        '''
        self.base, self.name = os.path.split(filename)
        self.filt_extra = filt_extra
        nfilts = 2
        if filters is None:
            _, filters = parse_pipeline(filename)
            try:
                self.filter1, self.filter2 = filters
            except:
                try:
                    self.filter1, self.filter2, self.filter3 = filters
                    nfilts = 3
                except:
                    self.filter1 = 'V'
                    self.filter2 = 'I'
        else:
            for i in range(len(filters)):
                self.__setattr__('filter{0:d}'.format(i+1), filters[i])
        self.nfilters = len(filters)
        self.read_file(filename)

    def recovered(self, threshold=9.99):
        '''
        find indicies of stars with magdiff < threshold

        Parameters
        ----------
        threshold: float
            [9.99] magin - magout threshold for recovery

        Returns
        -------
        self.rec: list
            recovered stars in both filters
        rec1, rec2: list, list
            recovered stars in filter1, filter2
        '''
        rec1, = np.nonzero(np.abs(self.mag1diff) < threshold)
        rec2, = np.nonzero(np.abs(self.mag2diff) < threshold)
        self.rec = list(set(rec1) & set(rec2))
        retv = (rec1, rec2)
        if len(self.rec) == len(self.mag1diff):
            logger.warning('all stars recovered')

        if self.nfilters == 4:
            rec3, = np.nonzero(np.abs(self.mag3diff) < threshold)
            rec4, = np.nonzero(np.abs(self.mag4diff) < threshold)
            self.rec2 = list(set(rec3) & set(rec4))
            retv = (rec1, rec2, rec3, rec4)
            if len(self.rec2) == len(self.mag3diff):
                logger.warning('all stars recovered')

        return retv

    def make_hess(self, binsize=0.1, yattr='mag2diff', hess_kw={},
                  yattr2='mag4diff'):
        '''make hess grid'''
        self.colordiff = self.mag1diff - self.mag2diff
        mag = self.__getattribute__(yattr)
        self.hess = astronomy_utils.hess(self.colordiff, mag, binsize,
                                         **hess_kw)
        if self.nfilters == 4:
            self.colordiff2 = self.mag3diff - self.mag4diff
            mag = self.__getattribute__(yattr2)
            self.hess = astronomy_utils.hess(self.colordiff2, mag, binsize,
                                             **hess_kw)

    def read_file(self, filename):
        '''
        read MATCH fake file into attributes
        format is mag1in mag1diff mag2in mag2diff
        mag1 is assumed to be mag1in
        mag2 is assumed to be mag2in
        mag1diff is assumed to be mag1in-mag1out
        mag2diff is assumed to be mag2in-mag2out
        '''
        if not filename.endswith('.fits'):
            with open(filename, 'r') as inp:
                cols = inp.readline().strip().split()
            nmags = len(cols) // 2
            mags = ['mag{:d}'.format(i+1) for i in range(nmags)]
            mdifs = ['{:s}diff'.format(m) for m in mags]
            names = list(np.concatenate([mags, mdifs]))
            self.data = np.genfromtxt(filename, names=names)
            # unpack into attribues
            for name in names:
                self.__setattr__(name, self.data[name])
        else:
            assert not None in [self.filter1, self.filter2], \
                'Must specify filter strings'
            self.data = fits.getdata(filename)
            self.mag1 = self.data['{}_IN'.format(self.filter1)]
            self.mag2 = self.data['{}_IN'.format(self.filter2)]
            mag1out = self.data['{}{}'.format(self.filter1, self.filt_extra)]
            mag2out = self.data['{}{}'.format(self.filter2, self.filt_extra)]
            self.mag1diff = self.mag1 - mag1out
            self.mag2diff = self.mag2 - mag2out

    def write_matchfake(self, newfile, filterset=0):
        '''write matchfake file'''
        if filterset == 1:
            dat = np.array([self.mag1, self.mag2, self.mag1diff, self.mag2diff]).T
        else:
            dat = np.array([self.mag3, self.mag4, self.mag3diff, self.mag4diff]).T

        np.savetxt(newfile, dat, fmt='%.3f')

    def bin_asts(self, binsize=0.2, bins=None, filterset=0):
        '''
        bin the artificial star tests

        Parameters
        ----------
        bins: bins for the asts
        binsize: width of bins for the asts

        Returns
        -------
        self.am1_inds, self.am2_inds: the indices of the bins to
            which each value in mag1 and mag2 belong (see np.digitize).
        self.ast_bins: bins used for the asts.
        '''
        if filterset == 0:
            mag1 = self.mag1
            mag2 = self.mag2
            ast_bins_atr = 'ast_bins'
            am1_inds_atr = 'am1_inds'
            am2_inds_atr = 'am2_inds'
        else:
            mag1 = self.mag3
            mag2 = self.mag4
            ast_bins_atr = 'ast_bins2'
            am1_inds_atr = 'am3_inds'
            am2_inds_atr = 'am4_inds'

        if hasattr(self, ast_bins_atr):
            return

        if bins is None:
            ast_max = np.max(np.concatenate((mag1, mag2)))
            ast_min = np.min(np.concatenate((mag1, mag2)))
            ast_bins = np.arange(ast_min, ast_max, binsize)
        else:
            ast_bins = bins

        self.__setattr__(am1_inds_atr, np.digitize(mag1, ast_bins))
        self.__setattr__(am2_inds_atr, np.digitize(mag2, ast_bins))
        self.__setattr__(ast_bins_atr, ast_bins)
        return

    def _random_select(self, arr, nselections):
        '''
        randomly sample arr nselections times

        Parameters
        ----------
        arr : array or list
            input to sample
        nselections : int
            number of times to sample

        Returns
        -------
        rands : array
            len(nselections) of randomly selected from arr (duplicates included)
        '''
        rands = np.array([np.random.choice(arr) for i in range(nselections)])
        return rands

    def ast_correction(self, obs_mag1, obs_mag2, binsize=0.2, bins=None,
                       not_rec_val=np.nan, missing_data1=0., missing_data2=0.,
                       filterset=0):
        '''
        Apply ast correction to input mags.

        Corrections are made by going through obs_mag1 in bins of
        bin_asts and randomly selecting magdiff values in that ast_bin.
        obs_mag2 simply follows along since it is tied to obs_mag1.

        Random selection was chosen because of the spatial nature of
        artificial star tests. If there are 400 asts in one mag bin,
        and 30 are not recovered, random selection should match the
        distribution (if there are many obs stars).

        If there are obs stars in a mag bin where there are no asts,
        will throw the star out unless the completeness in that mag bin
        is more than 50%.
        Parameters
        ----------
        obs_mag1, obs_mag2 : N, 1 arrays
            input observerd mags

        binsize, bins : sent to bin_asts

        not_rec_val : float
            value for not recovered ast
        missing_data1, missing_data2 : float, float
            value for data outside ast limits per filter (include=0)

        Returns
        -------
        cor_mag1, cor_mag2: array, array
            ast corrected magnitudes

        Raises:
            returns -1 if obs_mag1 and obs_mag2 are different sizes

        To do:
        Maybe not asssume combined_filters=True or completeness.
        A minor issue unless the depth of the individual filters are
        vastly different.
        '''
        self.completeness(combined_filters=True, interpolate=True,
                          filterset=filterset)

        nstars = obs_mag1.size
        if obs_mag1.size != obs_mag2.size:
            logger.error('mag arrays of different lengths')
            return -1

        # corrected mags are filled with nan.
        cor_mag1 = np.empty(nstars)
        cor_mag1.fill(not_rec_val)
        cor_mag2 = np.empty(nstars)
        cor_mag2.fill(not_rec_val)

        # need asts to be binned for this method.
        if not hasattr(self, 'ast_bins'):
            self.bin_asts(binsize=binsize, bins=bins, filterset=filterset)
        ast_bins_atr = 'ast_bins'
        am1_inds = 'am1_inds'
        fcomp2 = self.fcomp2
        mag1diff = self.mag1diff
        mag2diff = self.mag2diff
        if filterset == 1:
            ast_bins_atr += '2'
            am1_ins = 'am3_inds'
            fcomp2 = self.fcomp4
            mag1diff = self.mag3diff
            mag2diff = self.mag4diff

        am1_inds = self.__getattribute__(am1_ins)
        ast_bins = self.__getattribute__(ast_bins_atr)
        om1_inds = np.digitize(obs_mag1, ast_bins)

        for i in range(len(ast_bins)):
            # the obs and artificial stars in each bin
            obsbin, = np.nonzero(om1_inds == i)
            astbin, = np.nonzero(am1_inds == i)

            nobs = len(obsbin)
            nast = len(astbin)
            if nobs == 0:
                # no stars in this mag bin to correct
                continue
            if nast == 0:
                # no asts in this bin, probably means the simulation
                # is too deep
                if fcomp2(ast_bins[i]) < 0.5:
                    continue
                else:
                    # model is producing stars where there was no data.
                    # assign correction for missing data
                    cor1 = missing_data1
                    cor2 = missing_data2
            else:
                # randomly select the appropriate ast correction for obs stars
                # in this bin
                cor1 = self._random_select(mag1diff[astbin], nobs)
                cor2 = self._random_select(mag2diff[astbin], nobs)

            # apply corrections
            cor_mag1[obsbin] = obs_mag1[obsbin] + cor1
            cor_mag2[obsbin] = obs_mag2[obsbin] + cor2
            # finite values only: not implemented because trilegal array should
            # maintain the same size.
            #fin1, = np.nonzero(np.isfinite(cor_mag1))
            #fin2, = np.nonzero(np.isfinite(cor_mag2))
            #fin = list(set(fin1) & set(fin2))
        return cor_mag1, cor_mag2

    def correct(self, obs_mag1, obs_mag2, bins=[100,200], xrange=None,
                yrange=None, not_rec_val=0., dxy=None, filterset=0):
        """
        apply AST correction to obs_mag1 and obs_mag2

        Parameters
        ----------
        obs_mag1, obs_mag2 : arrays
            input mags to correct

        bins : [int, int]
            bins to pass to graphics.plotting.crazy_histogram2d

        xrange, yrange : shape 2, arrays
            limits of cmd space send to graphics.plotting.crazy_histogram2d
            since graphics.plotting.crazy_histogram2d is called twice it is
            important to have same bin sizes

        not_rec_val : float or nan
            value to fill output arrays where obs cmd does not overlap with
            ast cmd.

        dxy : array shape 2,
            color and mag step size to make graphics.plotting.crazy_histogram2d

        Returns
        -------
        cor_mag1, cor_mag2 : arrays len obs_mag1, obs_mag2
            corrections to obs_mag1 and obs_mag2
        """
        from ..utils.plotting_utils import crazy_histogram2d as chist

        nstars = obs_mag1.size
        if obs_mag1.size != obs_mag2.size:
            logger.error('mag arrays of different lengths')
            return -1, -1

        # corrected mags are filled with nan.
        cor_mag1 = np.empty(nstars)
        cor_mag1.fill(not_rec_val)
        cor_mag2 = np.empty(nstars)
        cor_mag2.fill(not_rec_val)

        obs_color = obs_mag1 - obs_mag2
        if filterset == 0:
            mag1 = self.mag1
            mag2 = self.mag2
            mag1diff = self.mag1diff
            mag2diff = self.mag2diff
        else:
            mag1 = self.mag3
            mag2 = self.mag4
            mag1diff = self.mag3diff
            mag2diff = self.mag4diff

        ast_color = mag1 - mag2

        if dxy is not None:
            # approx number of bins.
            if xrange is None:
                xmin = np.nanmin([np.nanmin(ast_color), np.nanmin(obs_color)])
                xmax = np.nanmax([np.nanmax(ast_color), np.nanmax(obs_color)])
            else:
                xmin, xmax = xrange
            if yrange is None:
                m2 = obs_mag2[obs_mag2 < 40.]
                am2 = mag2[mag2 < 40.]
                ymin = np.nanmin([np.nanmin(am2), np.nanmin(m2)])
                ymax = np.nanmax([np.nanmax(am2), np.nanmax(m2)])
            else:
                ymin, ymax = yrange

            bins[0] = len(np.arange(xmin, xmax, step=dxy[0]))
            bins[1] = len(np.arange(ymin, ymax, step=dxy[1]))

        ckw = {'bins': bins, 'reverse_indices': True, 'xrange': xrange,
               'yrange': yrange}

        SH, _, _, sixy, sinds = chist(ast_color,mag2, **ckw)
        H, _, _, ixy, inds = chist(obs_color, obs_mag2, **ckw)

        x, y = np.nonzero(SH * H > 0)
        # there is a way to do this with masking ...
        for i, j in zip(x, y):
            sind, = np.nonzero((sixy[:, 0] == i) & (sixy[:, 1] == j))
            hind, = np.nonzero((ixy[:, 0] == i) & (ixy[:, 1] == j))
            nobs = int(H[i, j])
            xinds = self._random_select(sinds[sind], nobs)
            cor_mag1[inds[hind]] = mag1diff[xinds]
            cor_mag2[inds[hind]] = mag2diff[xinds]

        return obs_mag1 + cor_mag1, obs_mag2 + cor_mag2

    def completeness(self, combined_filters=False, interpolate=False,
                     binsize=0.2, filterset=0):
        '''
        calculate the completeness of the data in each filter

        Parameters
        ----------
        combined_filters : bool
            Use individual or combined ast recovery

        interpolate : bool
            add a 1d spline the completeness function to self

        Returns
        -------
        self.comp1, self.comp2 : array, array
            the completeness per filter binned with self.ast_bins
        '''
        # calculate stars recovered, could pass theshold here.
        if filterset == 0:
            rec1, rec2 = self.recovered()
        else:
            _, _, rec1, rec2 = self.recovered()
        # make sure ast_bins are good to go
        self.bin_asts(binsize=binsize, filterset=filterset)

        # gst uses both filters for recovery.
        if combined_filters is True:
            if filterset == 0:
                rec1 = rec2 = self.rec
            else:
                rec1 = rec2 = self.rec2

        if filterset == 0:
            mag1 = self.mag1
            mag2 = self.mag2
            ast_bins = self.ast_bins
            comp1_atr = 'comp1'
            comp2_atr = 'comp2'
        else:
            mag1 = self.mag3
            mag2 = self.mag4
            ast_bins = self.ast_bins2
            comp1_atr = 'comp3'
            comp2_atr = 'comp4'

        # historgram of all artificial stars
        qhist1 = np.array(np.histogram(mag1, bins=ast_bins)[0], dtype=float)

        # histogram of recovered artificial stars
        rhist1 = np.array(np.histogram(mag1[rec1], bins=ast_bins)[0],
                          dtype=float)

        # completeness histogram
        comp1 = rhist1 / qhist1
        self.__setattr__(comp1_atr, comp1)

        qhist2 = np.array(np.histogram(mag2, bins=ast_bins)[0],
                          dtype=float)
        rhist2 = np.array(np.histogram(mag2[rec2], bins=ast_bins)[0],
                          dtype=float)

        comp2 = rhist2 / qhist2
        self.__setattr__(comp2_atr, comp2)

        if interpolate is True:
            # sometimes the histogram isn't as useful as the a spline
            # function... add the interp1d function to self.
            self.__setattr__('f{0:s}'.format(comp1_atr),
                             interp1d(ast_bins[1:], comp1, bounds_error=False))
            self.__setattr__('f{0:s}'.format(comp2_atr),
                             interp1d(ast_bins[1:], comp2, bounds_error=False))
        return

    def get_completeness_fraction(self, frac, dmag=0.001, bright_lim=18,
                                  filterset=0):
        """Find the completeness magnitude at a given fraction"""
        if filterset == 0:
            fcomp1_atr = 'fcomp1'
            fcomp2_atr = 'fcomp2'
        else:
            fcomp1_atr = 'fcomp3'
            fcomp1_atr = 'fcomp4'

        assert hasattr(self, fcomp1_atr), \
            'need to run completeness with interpolate=True'

        fcomp1 = self.__getattribute__(fcomp1_atr)
        fcomp2 = self.__getattribute__(fcomp2_atr)
        # set up array to evaluate interpolation
        # sometimes with few asts at bright mags the curve starts with low
        # completeness, reaches toward 1, and then declines as expected.
        # To get around taking a value too bright, I search for values beginning
        # at the faint end
        search_arr = np.arange(bright_lim, 31, dmag)[::-1]

        # completeness in each filter, and the finite vals
        # (frac - nan = frac)
        cfrac1 = fcomp1(search_arr)
        ifin1 = np.isfinite(cfrac1)

        cfrac2 = fcomp2(search_arr)
        ifin2 = np.isfinite(cfrac2)

        # closest completeness fraction to passed fraction
        icomp1 = np.argmin(np.abs(frac - cfrac1[ifin1]))
        icomp2 = np.argmin(np.abs(frac - cfrac2[ifin2]))

        # mag associated with completeness
        comp1 = search_arr[ifin1][icomp1]
        comp2 = search_arr[ifin2][icomp2]

        if comp1 == bright_lim or comp2 == bright_lim:
            logger.warning('Completeness fraction is at mag search limit and probably wrong. '
                           'Try adjusting bright_lim')
        return comp1, comp2

    def magdiff_plot(self, axs=None, filterset=0):
        """Make a plot of input mag - output mag vs input mag"""
        if not hasattr(self, 'rec'):
            self.completeness(combined_filters=True, filterset=filterset)
        xlab = r'${{\rm Input}}\ {}$'
        if axs is None:
            fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

        if filterset == 0:
            axs[0].plot(self.mag1[self.rec], self.mag1diff[self.rec], '.',
                        color='k', alpha=0.5)
            axs[1].plot(self.mag2[self.rec], self.mag2diff[self.rec], '.',
                        color='k', alpha=0.5)
            axs[0].set_xlabel(xlab.format(self.filter1), fontsize=20)
            axs[1].set_xlabel(xlab.format(self.filter2), fontsize=20)
        else:
            axs[0].plot(self.mag3[self.rec], self.mag3diff[self.rec], '.',
                        color='k', alpha=0.5)
            axs[1].plot(self.mag4[self.rec2], self.mag4diff[self.rec2], '.',
                        color='k', alpha=0.5)
            axs[0].set_xlabel(xlab.format(self.filter3), fontsize=20)
            axs[1].set_xlabel(xlab.format(self.filter4), fontsize=20)

        axs[0].set_ylabel(r'${{\rm Input}} - {{\rm Ouput}}$', fontsize=20)
        return axs

    def completeness_plot(self, ax=None, comp_fracs=None):
        """Make a plot of completeness vs mag"""
        assert hasattr(self, 'fcomp1'), \
            'need to run completeness with interpolate=True'

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.ast_bins, self.fcomp1(self.ast_bins),
                label=r'${}$'.format(self.filter1))
        ax.plot(self.ast_bins, self.fcomp2(self.ast_bins),
                label=r'${}$'.format(self.filter2))

        if comp_fracs is not None:
            self.add_complines(ax, *comp_fracs)
        ax.set_xlabel(r'${{\rm mag}}$', fontsize=20)
        ax.set_ylabel(r'${{\rm Completeness\ Fraction}}$', fontsize=20)
        plt.legend(loc='lower left', frameon=False)
        return ax

    def add_complines(self, ax, *fracs, **get_comp_frac_kw):
        """add verticle lines to a plot at given completeness fractions"""
        lblfmt = r'${frac}\ {filt}:\ {comp: .2f}$'
        for frac in fracs:
            ax.axhline(frac, alpha=0.5)
            comp1, comp2 = self.get_completeness_fraction(frac,
                                                          **get_comp_frac_kw)
            for comp, filt in zip((comp1, comp2), (self.filter1, self.filter2)):
                lab = lblfmt.format(frac=frac, filt=filt, comp=comp)
                ax.axvline(comp, label=lab,
                          color=next(ax._get_lines.color_cycle))
        plt.legend(loc='lower left', frameon=False)
        return ax


def main(argv):
    parser = argparse.ArgumentParser(description="Calculate completeness fraction, make AST plots")

    parser.add_argument('-c', '--comp_frac', type=float, default=0.9,
                        help='completeness fraction to calculate')

    parser.add_argument('-p', '--makeplots', action='store_true',
                        help='make AST plots')

    parser.add_argument('-m', '--bright_mag', type=float, default=20.,
                        help='brighest mag to consider for completeness frac')

    parser.add_argument('-f', '--plot_fracs', type=str, default=None,
                        help='comma separated completeness fractions to overplot')

    parser.add_argument('fake', type=str, nargs='*', help='match AST file(s)')

    args = parser.parse_args(argv)
    for fake in args.fake:
        ast = ASTs(fake)
        ast.completeness(combined_filters=True, interpolate=True,
                         binsize=0.15)
        comp1, comp2 = ast.get_completeness_fraction(args.comp_frac,
                                                     bright_lim=args.bright_mag)
        print('{} {} completeness fraction:'.format(fake, args.comp_frac))
        print('{0:20s} {1:.4f} {2:.4f}'.format(ast.name, comp1, comp2))

        if args.makeplots:
            comp_name = os.path.join(ast.base, ast.name + '_comp.png')
            ast_name = os.path.join(ast.base, ast.name + '_ast.png')

            ax = ast.completeness_plot()
            if args.plot_fracs is not None:
                fracs = map(float, args.plot_fracs.split(','))
                ast.add_complines(ax, *fracs, **{'bright_lim': args.bright_mag})
            plt.savefig(comp_name)
            plt.close()

            ast.magdiff_plot()
            plt.savefig(ast_name)
            plt.close()


if __name__ == "__main__":
    main(sys.argv[1:])
