"""
Double gaussian contamination!
"""
import argparse
import sys
import numpy as np
import os
import ResolvedStellarPops as rsp
from ResolvedStellarPops import utils
from astropy.io import fits
from astropy.table import Table
from TPAGBparams import EXT, data_loc, snap_src
import difflib
from .fileio import get_files
from plotting.plotting import emboss
from astroML.plotting import hist as mlhist
from scipy import integrate
angst_data = rsp.angst_tables.angst_data

import logging
logger = logging.getLogger()

import matplotlib.pyplot as plt
plt.style.use('presentation')

def test_line(mag, off=0.0):
    return tpagb_rheb_line(mag, off=off)

def tpagb_rheb_line(mag, b=-7.2, m=-9.03226, dmod=0., Av=0.0, off=0.0):
    """
    Default values found in this code using UGC5139 and NGC3741
    b=6.21106, m=-8.97165
    by eye:
    b=1.17303, m=-5.20269
    median:
    b=6.653127, m=-9.03226
    set dmod and Av to 0 for absmag
    adjusted from median by eye:
    b=7.453127, m=-9.03226
    with off = f814w-f160w trgb
    b=-7, m=-9.03226
    """
    ah = 0.20443
    ai = 0.60559
    c = (ah + m * (ah - ai))
    return (mag - b - dmod - c * Av) / m + off


def get_itpagb(target, color, mag, col, blue_cut=-99, absmag=False,
               mtrgb=None, dmod=0.0, Av=0.0, off=0):
    # careful! get_snap assumes F160W
    if '160' in col or '110' in col or 'IR' in col:
        if mtrgb is None:
            try:
                mtrgb, Av, dmod = angst_data.get_snap_trgb_av_dmod(target.upper())
            except:
                logger.error('Target not found: get_snap_trgb_av_dmod {}'.format(target))
                return [np.nan] * 2
            if absmag:
                mtrgb =  rsp.astronomy_utils.mag2Mag(mtrgb, 'F160W', 'wfc3ir',
                                                     dmod=dmod, Av=Av)
                dmod = 0.
                Av = 0.
        cs = tpagb_rheb_line(mag, dmod=dmod, Av=Av, off=off)
        redward_of_rheb, = np.nonzero(color > cs)
        blueward_of_rheb, = np.nonzero(color < cs)

    else:
        logger.warning('Not using TP-AGB RHeB line')
        if mtrgb is None:
            mtrgb, Av, dmod = angst_data.get_tab5_trgb_av_dmod(target.upper(),
                                                               filters=col)
            if absmag:
                mtrgb = rsp.astronomy_utils.mag2Mag(mtrgb, col, 'acs_wfc',
                                                    dmod=dmod, Av=Av)
        redward_of_rheb = np.arange(len(color))
        blueward_of_rheb = np.arange(len(color))

    redward_of_bluecut, = np.nonzero(color > blue_cut)
    brighter_than_trgb, = np.nonzero(mag < mtrgb)
    itpagb = list(set(redward_of_rheb) & set(brighter_than_trgb))
    irheb = list(set(brighter_than_trgb) & set(redward_of_bluecut) & set(blueward_of_rheb))
    return itpagb, irheb

def deviations(x):
    return np.abs(x - np.mean(x)) / np.std(x)

class Contamination(object):
    def __init__(self):
        pass

    def fit_double_gaussian(self, counts, cbins, err=None):

        if err is None:
            # uniform errors
            err = np.zeros(len(cbins[:-1])) + 1.

        # set up inputs
        hist_in = {'x': cbins[:-1], 'y': counts, 'err': err}

        # set up initial parameters:
        # mean of gaussians is set to the two highest peaks from color histogram
        iextr = utils.find_peaks(counts)['maxima_locations']
        if len(iextr) < 2:

            logger.error('found {} peaks.'.format(len(iextr)))
            return np.nan
        maxofmax = iextr[np.argmax(counts[iextr])]
        g1mean = cbins[maxofmax]
        # get rid of that value
        iextr.pop(iextr.index(maxofmax))
        g2mean = cbins[iextr[np.argmax(counts[iextr])]]

        if g1mean > g2mean:
            g1mean, g2mean = g2mean, g1mean

        logger.debug('astroML hist peaks: {:.2f} {:.2f}'.format(g1mean, g2mean))
        #the old way:
        #g1mean = np.mean(cbins[1:]) - np.mean(cbins[1:]) / 2.
        #g2mean = np.mean(cbins[1:]) + np.mean(cbins[1:]) / 2.

        p0 = [np.nanmax(counts) / 2., g1mean, np.mean(np.diff(cbins)),
              np.nanmax(counts) / 2., g2mean, np.mean(np.diff(cbins))]

        # double gaussian fitting
        mp_dg = utils.mpfit(utils.mp_double_gauss, p0, functkw=hist_in, quiet=True)
        # single gaussiamn fitting
        mp_g = utils.mpfit(utils.mp_gauss, p0[:3], functkw=hist_in, quiet=True)

        # how'd we do...
        if mp_dg.covar is None:
            logger.error('Not double gaussian {}'.format(mp_dg.params))
            return np.nan
        else:
            perc_err = (mp_dg.perror - mp_dg.params) / mp_dg.params
            tot_err = np.sum([p ** 2 for p in perc_err])
            # arbitrary error
            if tot_err > 10.:
                logger.error('Not double guassian, errors too large {}'.format(tot_err))
                return np.nan
            perr = (mp_g.perror - mp_g.params) / mp_g.params
            terr = np.sum([p ** 2 for p in perr])
            if terr < tot_err:
                logger.warning('Less error as a single gaussian {}'.format(tot_err / terr))
        return mp_dg

    def extract_gaussians(self, mp_dg, carr):
        g_p1 = mp_dg.params[0: 3]
        g_p2 = mp_dg.params[3:]
        # sort the gaussians
        if g_p1[1] > g_p2[1]:
            g_p2, g_p1 = g_p1, g_p2
        gauss1 = utils.gaussian(carr, g_p1)
        gauss2 = utils.gaussian(carr, g_p2)

        return g_p1, gauss1, g_p2, gauss2

    def find_contamination(self, mp_dg, cbins, nstars, color_sep=None):
        rval = np.nan, [np.nan] * 6
        # take fit params and apply to guassians on an arb color scale
        carr = np.linspace(cbins[0], cbins[-1], 1000)
        g_p1, gauss1, g_p2, gauss2 = self.extract_gaussians(mp_dg, carr)

        logger.info('mpfit peaks: {:.2f} {:.2f}'.format(g_p1[1], g_p2[1]))

        dcol = np.mean(np.diff(cbins))

        if cbins[0] + dcol > g_p1[1]:
            logger.error('First gaussian too close to edge')
            return rval

        if cbins[-1] - dcol < g_p2[1]:
            logger.error('Second gaussian too close to edge')
            return rval

        ginds, = np.nonzero((carr > g_p1[1]) & (carr < g_p2[1]))

        # color separatrion is the intersection of the two gaussians..
        min_locs = np.argmin(np.abs(gauss1[ginds] - gauss2[ginds]))
        auto_color_sep = carr[ginds][min_locs]

        logger.info('Estimated seperation color: {}'.format(auto_color_sep))

        if auto_color_sep == carr[0]:
            auto_color_sep = np.mean([g_p1[1], g_p2[1]])
            logger.error('using mean between hist peaks as color_sep')

        if color_sep is None:
            color_sep = auto_color_sep
        else:
            logger.info('Input color_sep: {:.4f} found it at {:.4f}'.format(color_sep,
                                                                      auto_color_sep))

        # find contamination past the color sep...
        g12_integral = integrate.quad(utils.double_gaussian, -np.inf, np.inf,
                                      mp_dg.params)[0]
        try:
            norm = nstars / g12_integral
        except ZeroDivisionError:
            logger.error('Double gaussian integral is zero')
            return rval

        # integral of each gaussian
        g1_integral = integrate.quad(utils.gaussian, -np.inf, np.inf, g_p1)[0]
        g2_integral = integrate.quad(utils.gaussian, -np.inf, np.inf, g_p2)[0]

        # integrals from (-inf, colsep) to (colsep, inf)

        # most of the g1 integral (we hope)
        g1_colsep = integrate.quad(utils.gaussian, -np.inf, color_sep, g_p1)[0]
        # blead over from g1 passed colsep
        colsep_g1 = integrate.quad(utils.gaussian, color_sep, np.inf, g_p1)[0]

        # most of the g2 integral (we hope)
        colsep_g2 = integrate.quad(utils.gaussian, color_sep, np.inf, g_p2)[0]
        # blead over from g2 before colsep
        g2_colsep = integrate.quad(utils.gaussian, -np.inf, color_sep, g_p2)[0]

        # how'd we do...
        # integral of the "blead overs"
        lir = colsep_g1 * norm
        ril = g2_colsep * norm

        # fraction of stars from the (l, r) gaussian in the (r, l) gaussian
        # compared to total stars in the (l ,r) gaussian
        # or: how much of a difference does the bleading over make to the donor
        try:
            lirpl = colsep_g1 / g1_integral
            rilpr = g2_colsep / g2_integral
        except ZeroDivisionError:
            logger.error('Single gaussian integral is zero')
            return rval
        # fraction of stars from the (l, r) gaussian in the (r, l) gaussian
        # compared to total stars in the (r ,l)
        # or: how much of a difference does the bleading over make to the recipient
        try:
            lirpr = colsep_g1 / colsep_g2
            rilpl = g2_colsep / g1_colsep
        except ZeroDivisionError:
            logger.error('Single gaussian integral is zero')
            return rval

        return color_sep, [lir, ril, lirpl, lirpr, rilpr, rilpl]

    def double_gaussian_intersection(self, color, mag, verts=None, test=None,
                                     diag_plot=False, thresh=5, trgb_color=0.0):
        '''
        This function fits a double gaussian to a color histogram of stars
        within the <maglimits> and <colorlimits> (tuples).

        It then finds the intersection of the two gaussians, and the fraction
        of each integrated gaussian that crosses over the intersection color
        line.
        Parameters
        ----------
        color : arr
            xaxis values
        mag : arr
            yaxis values
        verts : Nx2 arr
            polygon verticies of [color, mag] of the paramater space
        color_sep : float
            user estimated color separation
        diag_plot : bool
            make a plot of the resultant gaussians
        thresh : int
            min number of stars to do the fitting (useful if this is run iterativly)

        Returns
        -------

        '''
        # more like error val, something to return.
        rval = np.nan, np.nan, []

        if verts is not None:
            points = np.column_stack((color, mag))
            inds, = np.nonzero(utils.points_inside_poly(points, verts))

        else:
            inds = np.arange(len(color))

        if len(inds) <= thresh:
            logger.error('Not enough points found within verts')
            return rval
        # make a color histogram
        col = color[inds]
        mean_mag = np.mean(mag[inds])

        # don't be clever with bins:
        #dcol = 0.05
        #cbins = np.arange(col.min(), col.max() + dcol, dcol)
        #counts = np.histogram(col, bins=cbins)[0]

        # be clever with bins (why does mlhist have to call gca?):
        figx = plt.subplots()
        counts, cbins, _ = mlhist(col, bins='knuth')
        plt.close()
        dcol = np.diff(cbins)
        if np.mean(dcol) > 0.2:
            logger.error('Color bins are too wide')
            return rval

        mp_dg = self.fit_double_gaussian(counts, cbins)
        if not hasattr(mp_dg, 'params'):
            return rval

        if test is None:
            color_sep = None
        else:
            color_sep = test_line(mean_mag, off=trgb_color)

        color_sep, [lir, ril, lirpl, lirpr, rilpr, rilpl] = \
            self.find_contamination(mp_dg, cbins, float(len(inds)),
                                    color_sep=color_sep)
        if np.isnan(color_sep):
            return rval

        if lirpr > 0.15 or rilpr > 0.15:
            logger.error('contamination is too high to use as a data point')
            return rval

        fmt = '{0} in {1}: {2:.0f} {3:.2f}\% {0} which is {4:.2f}\% {1}'
        if diag_plot:
            carr = np.linspace(cbins[0], cbins[-1], 1000)
            g_p1, gauss1, g_p2, gauss2 = self.extract_gaussians(mp_dg, carr)

            fig, ax = plt.subplots()
            ax.plot(cbins[:-1], counts, ls='steps', label='input')
            ax.plot(carr, utils.double_gaussian(carr, mp_dg.params),
                    label='double gaussian')
            ax.plot(carr, gauss1, label='gaussian 1')
            ax.plot(carr, gauss2, label='gaussian 2')
            ax.legend(loc='best')
            ax.set_xlim(col.min(), col.max())
            ax.set_xlabel('$color$', fontsize=20)
            ax.set_ylabel('$\#$', fontsize=20)
            ax.set_title('Mean mag: {:.2f}'.format(mean_mag))
            ax.vlines(color_sep, *ax.get_ylim())
            fmt = '{0} in {1}: {2:.0f} {3:.2f}\% {0} which is {4:.2f}\% {1}'
            ax.text(0.1, 0.95, fmt.format('left', 'right', lir, lirpl, lirpr),
                    transform=ax.transAxes, fontsize=10)
            ax.text(0.1, 0.90, fmt.format('right', 'left', ril, rilpr, rilpl),
                    transform=ax.transAxes, fontsize=10)

        logger.info(fmt.format('left', 'right', lir, lirpl, lirpr))
        logger.info(fmt.format('right', 'left', ril, rilpr, rilpl))
        return color_sep, mp_dg, [lir, ril, lirpl, lirpr, rilpr, rilpl]

    def mag_steps(self, color, mag, verts=None, test=None, diag_plot=False,
                  thresh=20, threshlimit=True, nanlimit=True, bins='knuth',
                  trgb_color=0.0):
        if verts is not None:
            points = np.column_stack((color, mag))
            inds, = np.nonzero(utils.points_inside_poly(points, verts))
            clim = [verts[0, 0], verts[-2, 0]]
        else:
            inds = np.arange(len(color))
            clim = [color.min(), color.max()]

        smag = mag[inds]
        scolor = color[inds]
        spoints = np.column_stack((scolor, smag))

        figx = plt.subplots()
        counts, mbins, _ = mlhist(smag, bins=bins)
        plt.close()

        #dmag = 0.4
        #mbins = np.arange(smag.min(), smag.max() + dmag, dmag)
        #counts, mbins= np.histogram(smag, bins=mbins)

        mverts = []
        nstars = np.array([])
        i = len(mbins) - 1
        while i > 1:
            j = i - 1
            mvs = np.array([[clim[0], mbins[i]],
                            [clim[0], mbins[j]],
                            [clim[1], mbins[j]],
                            [clim[1], mbins[i]],
                            [clim[0], mbins[i]]])
            inds, = np.nonzero(utils.points_inside_poly(spoints, mvs))
            if threshlimit:
                # test if there are enough stars in the bin, if not,
                # increase the bin size
                while len(inds) < thresh:
                    j -= 1
                    if j < 0:
                        break
                    mvs[1:3, 1] = mbins[j]
                    inds, = np.nonzero(utils.points_inside_poly(spoints, mvs))
                i = j
                mverts.append(mvs)
            else:
                mverts.append(mvs)
                i -=1

        logger.debug('started with {} mag bins'.format(len(mbins)))
        if threshlimit:
            logger.debug('thresh limited to {} mag bins'.format(len(mverts)))

        color_seps = []
        if nanlimit:
            # test to see if double gaussian fails, if it does, increase
            # the bin size
            j = 0
            k = 0
            mdverts = []
            while j < len(mverts):
                k += 1
                if k == len(mverts) - 1:
                    mdverts.append(mverts[j])
                    break

                cs =  self.double_gaussian_intersection(scolor, smag, test=test,
                                                        verts=mverts[j],
                                                        trgb_color=trgb_color)[0]
                if np.isnan(cs):
                    mverts[j][1:3, 1] = mverts[k][1:3, 1]
                else:
                    mdverts.append(mverts[j])
                    j += 1
            logger.debug('nan limited to {} mag bins'.format(len(mdverts)))
        else:
            mdverts = mverts

        cs = np.array([])
        ms = np.array([])
        carr = np.linspace(scolor.min(), scolor.max(), 10000)
        fig, ax = plt.subplots()
        ax.scatter(color, mag, marker='.', s=8, c='k', alpha=1)
        ax.set_ylim(-3, -10)
        ax.set_xlim(-1, 4)
        self.result = {}
        self.mverts = mdverts
        for i, mvert in enumerate(mdverts):

            low = mvert[0, 1]
            dmag = mvert[0, 1] - mvert[1, 1]
            color_sep, mp_dg, result = \
                self.double_gaussian_intersection(scolor, smag, verts=mvert,
                                                  test=test, trgb_color=trgb_color,
                                                  diag_plot=diag_plot)
            self.result['{}'.format(i)] = result

            if np.isfinite(color_sep):
                lir, ril, lirpl, lirpr, rilpr, rilpl = result
                double_gauss = utils.double_gaussian(carr, mp_dg.params)
                norm = -1 * dmag / np.max(double_gauss)
                ax.plot(mvert[:, 0], mvert[:, 1], color='k', alpha=0.5, lw=1)
                ax.plot(carr, double_gauss * norm + low, lw=2, color='w')
                ax.plot(carr, double_gauss * norm + low, lw=1, color='k')
                #ax.plot(color_sep, np.mean(mvert[:, 1][:-1]), 'o', color='r')
                ax.text(scolor.max() + 0.02, low, r'${:.2f}$'.format(rilpr),
                        fontsize=14)
                ax.text(scolor.min() - 0.02, low, r'${:.2f}$'.format(lirpl),
                        fontsize=14, ha='right')

                cs = np.append(cs, color_sep)
                ms = np.append(ms, np.mean(np.array(mvert)[:, 1][:-1]))

        if len(cs) >= 2 and test is None:
            inds, = np.nonzero(deviations(cs) < 1.68)
            f = np.polyfit(cs[inds], ms[inds], 1)
            xarr = np.arange(verts[0, 0], verts[-2, 0], 0.01)
            yarr = f[0] * xarr + f[1]
            inds, = np.nonzero((yarr <= verts[0, 1]) & (yarr >= verts[1, 1]))
            #ax.plot(xarr[inds], yarr[inds], color='k')
        else:
            f = np.array([0.0, 0.0])
            plt.close()
        return cs, ms, f, ax, fig


def load_data(fname, absmag=False):
    gal = rsp.galaxy.Galaxy(fname)
    mtrgb, Av, dmod = gal.trgb_av_dmod('F160W')
    verts = np.array([[1, mtrgb], [1, 19], [3, 19], [3, mtrgb], [1, mtrgb]])
    mag = gal.data['MAG4_IR']
    try:
        color = gal.data['MAG2_ACS'] - mag
    except:
        logger.error('WFPC2 not supported')
        return [], [], [], np.nan
    if absmag:
        mtrgb = rsp.astronomy_utils.mag2Mag(mtrgb, 'F160W', 'wfc3ir', dmod=dmod, Av=Av)
        verts[1:3, 1] = -11
        verts[(0,3,4), 1] = mtrgb
        mag = gal.absmag('MAG4_IR', 'F160W', photsys='wfc3ir', dmod=dmod, Av=Av)
        mag1 = gal.absmag('MAG2_ACS', 'F814W', photsys='acs_wfc', dmod=dmod, Av=Av)
        color = mag1 - mag
    return color, mag, verts, mtrgb


def test_contamination_line(filenames, diag_plot=False):
    ftpinrhebs = np.array([])
    frhebintps= np.array([])
    nrhebcontams = np.array([])
    ntpcontams = np.array([])
    for fname in filenames:
        # load trilegal simulation
        sgal = rsp.SimGalaxy(fname)
        # find tpagb and rgb stars using the line
        try:
            mag1 = sgal.data['F814W_cor']
            mag2 = sgal.data['F160W_cor']
        except:
            logger.error('missing ast corrections: {}'.format(fname))
            continue
        good, = np.nonzero((np.abs(mag1) < 30) & (np.abs(mag2) < 30))
        color = mag1[good] - mag2[good]
        mag = mag2[good]
        stage = sgal.data['stage'][good]

        dmod = sgal.data['m-M0'][0]
        Av = sgal.data['Av'][0]

        mtrgb = angst_data.get_snap_trgb_av_dmod(sgal.target.upper())[0]
        opt_mtrgb = angst_data.get_tab5_trgb_av_dmod(sgal.target.upper())[0]
        trgb_color = opt_mtrgb - mtrgb

        # also eagb (7)
        #itpagb, = np.nonzero((stage == 8) | (stage == 7))
        itpagb, = np.nonzero((stage == 8))
        # all HeB
        irheb, = np.nonzero((stage >= 4) & (stage <= 6))
        itpagbx, irhebx = get_itpagb(sgal.target, color, mag, 'F160W', off=trgb_color)

        nrheb = float(len(irhebx))
        ntpagb = float(len(itpagbx))

        _, itp_in_rheb = get_itpagb(sgal.target, color[itpagb], mag[itpagb],
                                    'F160W', mtrgb=mtrgb, dmod=dmod, Av=Av, off=trgb_color)
        irheb_in_tp, _ = get_itpagb(sgal.target, color[irheb], mag[irheb],
                                    'F160W', mtrgb=mtrgb, dmod=dmod, Av=Av, off=trgb_color)

        ntpcontam = float(len(itp_in_rheb))
        nrhebcontam = float(len(irheb_in_tp))
        try:
            ftpinrheb = ntpcontam / ntpagb
        except ZeroDivisionError:
            print 'No RHeB stars {}'.format(sgal.target)
            ftpinrheb = np.nan
        try:
            frhebintp = nrhebcontam / ntpagb
        except ZeroDivisionError:
            print 'No TP-AGB stars {}'.format(sgal.target)
            frhebintp = np.nan

        fmt = '{} {} stars in {} region {:.2f} of total TP-AGB on the TP-AGB side'
        logger.debug(sgal.target)
        logger.debug(fmt.format(ntpcontam, 'TP-AGB', 'RHeB', ftpinrheb))
        logger.debug(fmt.format(nrhebcontam, 'RHeB', 'TP-AGB', frhebintp))
        if diag_plot:
            x = tpagb_rheb_line(mag[mag < mtrgb], dmod=dmod, Av=Av, off=trgb_color)
            fig, ax = plt.subplots()
            ax.plot(color, mag, '.', color='k', alpha=0.2)
            ax.axhline(mtrgb)
            ax.plot(x, mag[mag < mtrgb])
            ax.plot(color[itpagb], mag[itpagb], '.', alpha=0.2, label='tpagb')
            ax.plot(color[irheb], mag[irheb], '.', alpha=0.2, label='rheb')
            ax.plot(color[itpagb[itp_in_rheb]], mag[itpagb[itp_in_rheb]], 'o', mec='none', label='tpagb in rheb')
            ax.plot(color[irheb[irheb_in_tp]], mag[irheb[irheb_in_tp]], 'o', mec='none', label='rheb in tpagb')
            ax.set_xlabel('$F814W-F160W$')
            ax.set_ylabel('$F160W$')
            ax.set_ylim(26, ax.get_ylim()[0])
            plt.legend(loc='best')
            plt.savefig(fname + '_contam{}'.format(EXT))
            plt.close()
        ftpinrhebs = np.append(ftpinrhebs, ftpinrheb)
        frhebintps = np.append(frhebintps, frhebintp)
        ntpcontams = np.append(ntpcontams, ntpcontam)
        nrhebcontams = np.append(nrhebcontam, nrhebcontam)

    logger.info('{:.2f} max tp in rhebs'.format(np.max(ftpinrhebs)))
    logger.info('{:.2f} mean tp in rhebs'.format(np.mean(ftpinrhebs)))
    logger.info('{:.2f} max rheb in tp'.format(np.max(frhebintps)))
    logger.info('{:.2f} mean rheb in tp'.format(np.mean(frhebintps)))

    return ftpinrhebs, frhebintps, ntpcontams, nrhebcontams


def find_data_contamination(fitsfiles, search=False, diag_plot=False, absmag=True,
                            threshlimit=True, nanlimit=True, result_plot=False,
                            off=0.0):
    """
    The test suite... to get the b, m to try.
    fitsfiles = !! ls *fits
    run once with test=None, then again with test_line when you got b,m to try
    """

    if search:
        logger.warning('overriding inputs')
        absmag = True
        from TPAGBparams import data_loc
        data_loc = os.path.join(data_loc, 'copy/')
        fitsfiles = rsp.fileio.get_files(data_loc, '*fits')
        plist = ['sn-ngc2403-pr_f606w_f814w_f110w_f160w.fits',
                 'ngc7793-halo-6_f606w_f814w_f110w_f160w.fits',
                 #'ngc404-deep_f606w_f814w_f110w_f160w.fits',
                 'ngc3077-phoenix_f555w_f814w_f110w_f160w.fits']
                 #'ugc8508_f475w_f814w_f110w_f160w.fits',
                 #'ugca292_f475w_f814w_f110w_f160w.fits',
                 #'ugc4305-2_f555w_f814w_f110w_f160w.fits',
                 #'ugc4305-1_f555w_f814w_f110w_f160w.fits',
                 #'ddo82_f606w_f814w_f110w_f160w.fits',
                 #'ngc4163_f606w_f814w_f110w_f160w.fits']

        [fitsfiles.pop(fitsfiles.index(p)) for p in plist if p in fitsfiles]
        test = None
        bins = 'knuth'
    else:
        test = test_line
        bins = 'knuth'
        #bins = 3
        #threshlimit = False

    filter1 = 'F814W'
    filter2 = 'F160W'
    xlabel = r'${}-{}$'.format(filter1, filter2)
    ylabel = r'${}$'.format(filter2)

    if result_plot:
        fig1, ax1 = plt.subplots()

    color_seps = []
    mean_mags = []
    fs = []
    y = []
    for fitsfile in fitsfiles:
        target = os.path.split(fitsfile)[1].split('_')[0]
        g = rsp.galaxy.Galaxy(fitsfile)
        trgb_color = g.trgb_av_dmod('F814W')[0] - g.trgb_av_dmod('F160W')[0]
        logger.debug(fitsfile)
        ctm = Contamination()
        color, mag, verts, mtrgb = load_data(fitsfile, absmag=absmag)
        if len(color) == 0:
            continue

        color_sep, mean_mag, f, ax, fig  = \
            ctm.mag_steps(color, mag, verts=verts, nanlimit=nanlimit,
                          diag_plot=diag_plot, threshlimit=threshlimit,
                          bins=bins, trgb_color=trgb_color)

        color_seps.append(color_sep)
        mean_mags.append(mean_mag)
        fs.append(f)

        #if len(color_sep) < 2:
        #    print('not enough color seps: {}'.format(color_sep))
        #    continue
        # result: lir, ril, lirpl, lirpr, rilpr,
        tpagbs = np.sum([ctm.result[k][1]/ctm.result[k][4] for k in ctm.result.keys() if len(ctm.result[k]) > 0])
        rhebs = np.sum([ctm.result[k][0]/ctm.result[k][-1] for k in ctm.result.keys() if len(ctm.result[k]) > 0])
        rheb_in_tpagb = np.sum([ctm.result[k][0] for k in ctm.result.keys() if len(ctm.result[k]) > 0])
        tpagb_in_rheb = np.sum([ctm.result[k][1] for k in ctm.result.keys() if len(ctm.result[k]) > 0])
        rheb_contam = rheb_in_tpagb / tpagbs
        tpagb_bleed = tpagb_in_rheb / tpagbs
        tpagb_contam = tpagb_in_rheb / rhebs
        logger.info('TOTAL CONTAMINATION RheB: {:.2f} TP-AGB contam {:.2f} TP-AGB bleed: {:.2f} {}'.format(rheb_contam, tpagb_contam, tpagb_bleed, fitsfile))

        if diag_plot:
            #ax.set_title(gal.target)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axhline(mtrgb, ls='--', color='k', lw=1)

            yarr = np.arange(*ax.get_xlim(), step=0.1)
            ax.fill_betweenx(yarr, mtrgb + 0.2, mtrgb - 0.2, color='k', alpha=0.2)
            ax.text(ax.get_xlim()[1] - 0.01, ax.get_ylim()[0] - 0.2, r'$\rm{{{}}}$'.format(target.replace('-', '\!-\!').upper()),
            fontsize=20, ha='right', **emboss())
            if test is not None:
                ax.plot(test(mag[mag < mtrgb], off=trgb_color), mag[mag < mtrgb], lw=2, color='k')
            fig.savefig('{}_{}-{}_contam{}'.format(target, filter1, filter2, EXT))

        if result_plot and f[0] > 0:
            xarr = np.arange(color.min(), color.max(), step=0.1)
            yarr = f[0] * xarr + f[1]
            ax1.plot(color, mag, ',', color='k', alpha=0.3)
            ax1.plot(color_sep, mean_mag, 'o', zorder=100, color='r', mec='none')
            ax1.plot(xarr, yarr, lw=2, label=target)

    cs = np.concatenate(color_seps)
    ms = np.concatenate(mean_mags)
    inds, = np.nonzero(deviations(cs) < 0.8)
    if len(inds) > 3:
        y = np.polyfit(cs[inds], ms[inds], 1)
    else:
        y = np.array([np.nan]) * 3

    if result_plot:
        xarr = np.arange(color.min(), color.max(), step=0.1)
        yarr = y[0] * xarr + y[1]
        ax1.plot(xarr, yarr, lw=3)
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.legend(loc='best')
        fig1.savefig('tp-agb_rheb_contam{}'.format(EXT))

    return color_seps, mean_mags, fs, y


def snappify(target):
    repl = {'GC': '',
            'UGC5139': 'HoI',
            'ESO540-030': 'KDG2',
            'U4305-1': 'HoII',
            'U4305-2': 'HoII',
            'U4459': 'DDO53',
            '-DEEP': '',
            '-HALO-6': '',
            'DDO71': 'KDG63',
            '-WIDE1': '',
            'SCL-DE1': 'Sc22'}
    for old, new in repl.iteritems():
        if old in target:
            target = target.replace(old, new)
    return target

def rgb_cut(full=False):
    def _add_text(ax, mb, trgb_color, target):
        # text placements
        ha = 'center'
        va = 'center'
        off = 0.015
        if '4459' in target or '300' in target:
            ha = 'left'
        if '8508' in target:
            ha = 'right'
        if '4305-2' in target:
            off = -0.01
            va = 'top'
        ax.text(mb, trgb_color + off,
                r'$\rm{{{}}}$'.format(target.replace('-', '\!-\!').upper()),
                ha=ha, va=va, fontsize=12)
        return ax

    fits_files = get_files(data_loc + '/copy', '*fits')

    if not full:
        targets = ['ESO540-030',
                   'KDG73',
                   'NGC2403',
                   'NGC3741',
                   'NGC4163',
                   'UGC4305-1',
                   'UGC4305-2',
                   'NGC300',
                   'UGC4459',
                   'UGC5139',
                   'UGC8508',
                   'UGCA292',
                   'DDO82']

        fits_files = np.concatenate([[s for s in fits_files if t.lower() in s]
                                     for t in targets])

    gals = [rsp.galaxy.Galaxy(f) for f in fits_files]
    fig, ax = plt.subplots()

    tab = Table.read(snap_src + '/tables/tab1.tex', format='latex',
                     data_start=1, delimiter='&', guess=False, header_start=0)
    line = '# target MB blue db trgb_color\n'
    for i, g in enumerate(gals):
        ir_mtrgb, Av, dmod = g.trgb_av_dmod('F160W')
        opt_mtrgb, Av, dmod = g.trgb_av_dmod('F814W')
        trgb_color = opt_mtrgb - ir_mtrgb
        target = g.name.split('_')[0].upper()
        targ1 = snappify(target)
        try:
            targ = difflib.get_close_matches(targ1, tab['Galaxy'])[0]
            mb = tab[tab['Galaxy'] == targ]['MB']
        except IndexError:
            print targ1, difflib.get_close_matches(targ1, tab['Galaxy'])
            mb = tab[tab['Galaxy'] == targ1]['MB']

        # blue edge or rgb-calibration box is 0.2 dex from the color
        blue = trgb_color - 0.2

        color = 'k'
        db = 0.4
        if mb <= -17.5:
            # brightest have wider RGBs.
            color = '#853E43'
            db = 0.6

        line += '{} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(target, float(mb), blue, db, trgb_color)

        ax.plot(mb, trgb_color, 'o', ms=9, color=color)
        ax = _add_text(ax, mb, trgb_color, target)

        contam_cmd(g, blue, ir_mtrgb, db, trgb_color, dmod, Av, target)

    ax.axvline(-17, color='k', ls='--')
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel(r'$M_{B_T}$')
    ax.set_ylabel(r'$\rm{(F814W-F160W)_{TRGB}}$')
    if not full:
        ax.set_xlim(-10, -20)
    fig.savefig('rgb_color_cut{}'.format(EXT))

    with open('rgb_color_cut.dat', 'w') as out:
        out.write(line)

    return

def contam_cmd(g, blue, ir_mtrgb, db, trgb_color, dmod, Av, target):
    fig1, ax1 = plt.subplots()
    try:
        color = g.data['MAG2_ACS'] - g.data['MAG4_IR']
        mag = g.data['MAG4_IR']
    except:
        color = g.data['MAG2_WFPC2'] - g.data['MAG4_IR']
        mag = g.data['MAG4_IR']

    ax1.plot(color, mag, 'o', ms=3, mec='none', c='k', alpha=0.5)

    verts = np.array([[blue, ir_mtrgb + 1],
                      [blue, ir_mtrgb],
                      [blue + db, ir_mtrgb],
                      [blue + db, ir_mtrgb + 1],
                      [blue, ir_mtrgb + 1]])
    ax1.plot(verts[:, 0], verts[:, 1], lw=2, color='#853E43')
    #ax1.axvline(trgb_color, color='#853E43')
    m = np.arange(mag.min(), ir_mtrgb, 0.1)
    cs = tpagb_rheb_line(m, dmod=dmod, Av=Av, off=trgb_color)
    ax1.plot(cs, m, color='k')
    itpagb = get_itpagb(target, color, mag, 'F160W', dmod=dmod, Av=Av,
                        off=trgb_color)[0]
    ax1.plot(color[itpagb], mag[itpagb], 'o', ms=5, mfc='none',
             mec='#156692', alpha=0.5)
    ax1.set_xlim(-0.5, 4)
    ax1.set_ylim(25.1, 17)
    ax1.text(ax1.get_xlim()[1] - 0.01,
             ax1.get_ylim()[0] - 0.2,
             r'$\rm{{{}}}$'.format(target.replace('-', '\!-\!').upper()),
             fontsize=20, ha='right', **emboss())

    ax1.set_xlabel(r'$\rm{F814W-F160W}$')
    ax1.set_ylabel(r'$\rm{F160W}$')
    fig1.savefig('{}_contam_cmd{}'.format(target, EXT))
    plt.close()


def main(argv):
    """Contaminatoin Tests"""
    description="Run contamination tests."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-a', '--absmag', action='store_false',
                        help='do not use absmag')

    parser.add_argument('-m', '--model', action='store_true',
                        help='run model contamination tests')

    parser.add_argument('-p', '--diag_plot', action='store_true',
                        help='make diag plots')

    parser.add_argument('-d', '--data', action='store_true',
                        help='run data contamination tests')

    parser.add_argument('-s', '--search', action='store_true',
                        help='with -d find the RHeB-TPAGB line')

    parser.add_argument('-r', '--result_plot', action='store_true',
                        help='with -d and -s make result plot')

    parser.add_argument('-f', '--rgb_cut', action='store_true',
                        help='make contam plots and trgb color vs MB')

    parser.add_argument('input_files', type=str, nargs='*',
                        help='fits files if -d trilegal simulatoins if -m')

    args = parser.parse_args(argv)

    logfile = 'contamination.log'
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.debug('command: {}'.format(' '.join(argv)))

    if args.model:
        test_contamination_line(args.input_files, diag_plot=args.diag_plot)
    elif args.data:
        find_data_contamination(args.input_files, search=args.search,
                                diag_plot=args.diag_plot, threshlimit=True,
                                nanlimit=True, result_plot=args.result_plot,
                                absmag=args.absmag)

if __name__ == '__main__':
    if '-f' in sys.argv[1:]:
        rgb_cut(full=False)
    else:
        main(sys.argv[1:])
