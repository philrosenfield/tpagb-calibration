import argparse
import difflib
import logging
import os
import sys

import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
sns.set()
sns.set_context('paper')
try:
    plt.style.use('paper')
except:
    pass

import numpy as np

from ..TPAGBparams import EXT, matchfake_loc, data_loc
from ..analysis.analyze import get_itpagb, parse_regions, get_trgb
from .. import fileio
from ..pop_synth import stellar_pops, SimGalaxy
from ..sfhs import star_formation_histories
from ..angst_tables import angst_data
from .. import utils
from ..utils import astronomy_utils
from ..utils.plotting_utils import emboss

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

tpagb_model_default_color = '#156692'
model_default_color = '#01141F'
data_tpagb_default_color = '#bc0003'
data_default_color = '#5e5e5e'

nar_text_fontsize = 18
gal_fontsize = 22
label_fontsize = 28
ticklabel_fontsize = 24


def add_narratio_to_plot(ax, target, ratio_data, mid_txt='RGB'):
    stext_kw = {'color': model_default_color, 'fontsize': nar_text_fontsize, 'ha': 'center'}
    stext_kw.update(emboss())
    dtext_kw = stext_kw
    dtext_kw.update({'color': data_default_color})

    assert ratio_data[0]['target'] == 'data', \
        'the first line of the narratio file needs to be a data measurement'
    nagb = float(ratio_data[0]['nagb'])
    nrgb = float(ratio_data[0]['nrgb'])

    dratio = nagb / nrgb
    dratio_err = utils.count_uncert_ratio(nagb, nrgb)

    indx, = np.nonzero(ratio_data['target'] == target)
    mrgb = np.mean(np.array(ratio_data[indx]['nrgb'], dtype=float))
    magb = np.mean(np.array(ratio_data[indx]['nagb'], dtype=float))

    stext_kw['transform'] = ax.transAxes
    dtext_kw['transform'] = ax.transAxes
    yval = 0.95
    xagb_val = 0.17
    xrgb_val = 0.5
    xratio_val = 0.83
    xvals = [xagb_val, xrgb_val, xratio_val]

    # simulated nrgb and nagb are the mean values
    srgb_text = r'$\langle N_{{\rm {0:s}}}\rangle ={1!s}$'.format(mid_txt, np.mean(mrgb))
    sagb_text = r'$\rm{{R14}}\ \langle N_{{\rm TP-AGB}}\rangle={0!s}}}$' % np.mean(magb)

    # one could argue taking the mean isn't the best idea for
    # the ratio errors.
    sratio_text = '$f=%.3f\pm%.3f$' % (np.mean(ratio_data['ar_ratio']),
                                       np.mean(ratio_data['ar_ratio_err']))

    drgb_text = r'$N_{\rm %s}=%i$' % (mid_txt, nrgb)
    dagb_text = r'$\rm{Data}\ N_{\rm TP-AGB}=%i$' % nagb
    dratio_text = '$f = %.3f\pm%.3f$' % (dratio, dratio_err)

    textss = [[sagb_text, srgb_text, sratio_text],
              [dagb_text, drgb_text, dratio_text]]
    kws = [stext_kw, dtext_kw]

    for kw, texts in zip(kws, textss):
        for xval, text in zip(xvals[::-1], texts[::-1]):
            if 'TP-AGB' in text:
                if 'langle' in text:
                    kw['color'] = tpagb_model_default_color
                else:
                    kw['color'] = data_tpagb_default_color
            ax.text(xval, yval, text, **kw)
        yval -= .05  # stack the text
    return ax


def plot_model(mag2s=None, bins=None, norms=None, inorm=None, ax=None,
               plt_kw={}, maglimit=None, agb_mod=None, mean=True,
               edgs=True):
    '''plot lf files
    Parameters
    ----------
    mag2s: list or array of arrays
        mag2 values from lf file

    bins: array
        bins used to make histogram of mag2s

    norms: list or array of floats
        normalize the histogram, will be overriden by inorm

    inorm: list or array of arrays
        indices of mag2 that sample the data distribution

    mag_limit:
        if specified will only plot mag <= maglimit

    agb_mod:
        if set, will add as legend label

    plt_kw:
        kwargs sent to ax.plot

    ax: axes instance
        if axes instance supplied, will not create new one

    Returns
    -------
    ax: axes instance
    '''
    default = {'color': model_default_color}
    default.update(plt_kw)
    plt_kw_lab = default
    if agb_mod is not None:
        label = '${}$'.format(agb_mod)
        plt_kw_lab.update({'label': label})

    if ax is None:
        fig, (ax) = plt.subplots(figsize=(10, 6))

    if inorm is not None:
        ms = [mag2s[i][inorm[i]] for i in range(len(mag2s))]
    else:
        ms =[mag2s[i] * norms[i] for i in range(len(mag2s))]

    hists = [np.histogram(m, bins=bins)[0] for m in ms]
    minhists = np.nanmin(np.array(hists).T, axis=1)
    maxhists = np.nanmax(np.array(hists).T, axis=1)
    #meanhists = np.mean(np.array(hists).T, axis=1)
    meanhists = np.nanmedian(np.array(hists).T, axis=1)
    edges = np.repeat(bins, 2)
    fminhist = np.hstack((0, np.repeat(minhists, 2), 0))
    fmaxhist = np.hstack((0, np.repeat(maxhists, 2), 0))

    ax.fill_between(edges, fminhist, fmaxhist, color=plt_kw_lab['color'],
                    alpha=0.2)

    #ax.fill_between(bins[1:], minhists, maxhists, color=plt_kw_lab['color'],
    #                alpha='0.2')
    if edgs:
        ax.plot(bins[1:], minhists, linestyle='steps', color=plt_kw_lab['color'],
                lw=0.8)
        ax.plot(bins[1:], maxhists, linestyle='steps', color=plt_kw_lab['color'],
                lw=0.8)
    l = None
    if mean:
        l, = ax.plot(bins[1:], meanhists, linestyle='steps', **plt_kw_lab)
    ylim = np.max(maxhists)
    return ax, l, ylim


def plot_gal(mag2, bins, ax=None, target=None, plot_kw={}, fake_file=None,
             over_plot=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    target = target.replace('-', '\!-\!').upper()
    default = {'drawstyle': 'steps', 'lw': 0.8, 'color': data_default_color,
               'label': '${}$'.format(target)}
    default.update(plot_kw)
    plot_kw = default

    hist = np.histogram(mag2, bins=bins)[0]
    err = np.sqrt(hist)

    if fake_file is not None:
        comp_corr = stellar_pops.completeness_corrections(fake_file, bins)
        hist = np.array(np.histogram(mag2, bins=bins)[0], dtype=float)
        comp_corr[np.isnan(comp_corr)] = 0.
        hist /= comp_corr[1:]
        err = np.sqrt(hist)
        plot_kw['lw'] += 0.8

    gl, = ax.plot(bins[1:], hist, **plot_kw)
    ylim = np.nanmax(hist)
    # mid bin
    ax.errorbar(bins[1:] - 0.05, hist, yerr=err, fmt='none',
                ecolor=plot_kw['color'], lw=plot_kw['lw']-0.1)
    glt = None
    if over_plot is not None:
        plot_kw['lw'] += 0.2
        hist = np.array(np.histogram(mag2[over_plot], bins=bins)[0], dtype=float)
        if fake_file is not None:
            hist /= comp_corr[1:]
        err = np.sqrt(hist)
        plot_kw['color'] = data_tpagb_default_color
        #plot_kw['label'] = '${}\ TP\!-\!AGB$'.format(target)
        plot_kw['label'] = '$TP\!-\!AGB$'.format(target)
        glt, = ax.plot(bins[1:], hist, **plot_kw)
        # mid bin
        ax.errorbar(bins[1:] - 0.05, hist, yerr=err, fmt='none',
                    ecolor=plot_kw['color'], lw=plot_kw['lw']-0.1)

    return ax, gl, glt, ylim


def plot_models(lfd, bins, filt, maglimit=None, ax=None, plt_kw=None,
                agb_mod=None):
    plt_kw = plt_kw or {}
    mags = lfd[filt]

    ax, ml, y = plot_model(mag2s=mags, bins=bins, inorm=lfd['idx_norm'], mean=False,
                        maglimit=maglimit, ax=ax, plt_kw=plt_kw, agb_mod=agb_mod,
                        edgs=False)

    plt_kw['color'] = tpagb_model_default_color
    ax, mlt, _ = plot_model(mag2s=mags, bins=bins, inorm=lfd['sim_agb'],
                        maglimit=maglimit, ax=ax, plt_kw=plt_kw,
                        agb_mod='TP\!-\!AGB')
                        #agb_mod='{}\ TP\!-\!AGB'.format(agb_mod))

    return ax, ml, mlt, y


def compare_to_gal(lf_file, observation, filter1='F814W_cor',
                   filter2='F160W_cor', col1='MAG2_ACS', col2='MAG4_IR',
                   dmag=0.1, narratio_file=None, make_plot=True,
                   regions_kw=None, xlims=[None, None], ylims=[None, None], extra_str='',
                   agb_mod=None, mplt_kw={}, dplot_kw={},
                   match_param=None, mtrgb=None, filterset=0,
                   mtrgb_mag2=None, fake=None):
    '''
    Plot the LFs and galaxy LF.

    ARGS:
    narratio: overlay NRGB, NAGB, and NAGB/NRGB +/- err
    no_agb: plot the LF without AGB stars

    RETURNS:
    ax1, ax2: axes instances created for the plot.

    '''
    trgb_color = 0.
    if mtrgb_mag2 is not None:
        trgb_color = mtrgb - mtrgb_mag2

    mag1, mag2 = fileio.load_observation(observation, col1, col2,
                                  match_param=match_param, filterset=filterset)

    data_tpagb = get_itpagb(mag1 - mag2, mag2, col2, off=trgb_color,
                            mtrgb=mtrgb)

    if 'cor' in filter1:
        if not '_ast_cor' in extra_str:
            extra_str += '_ast_cor'

    lfd = fileio.load_lf_file(lf_file)

    for filt, fake_file in zip([filter1, filter2], [optfake, nirfake]):
        if filt == filter1:
            mag = mag1
            xlim = xlims[0]
            ylim = ylims[0]
        else:
            mag = mag2
            xlim = xlims[1]
            ylim = ylims[1]

        bins = np.arange(mag.min(), mag.max(), step=dmag)

        ax, ml, mlt, ylm = plot_models(lfd, bins, filt, plt_kw=mplt_kw, agb_mod=agb_mod)

        # plot galaxy data
        ax, gl, glt, yld = plot_gal(mag, bins, ax=ax, target=target, fake_file=fake_file,
                      plot_kw=dplot_kw, over_plot=data_tpagb)

        ylim1 = np.max([ylm, yld])

        ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(1, np.round(ylim1 * 1e-3, 1) * 1e3)
        if xlim is not None:
            ax.set_xlim(xlim)

        band = 'nir'
        if '814' in filt:
            band = 'opt'
        ax = add_trgb(ax, target, band, regions_kw=regions_kw)

        #ax.legend(loc='center left', handles=[ml, gl, mlt, glt], ncol=2)
        lab = r'$\rm{{{}}}$'.format(target.upper().replace('-','\!-\!'))
        ax.text(0.98, 0.05, lab, ha='right', fontsize=gal_fontsize, transform=ax.transAxes,
                **emboss())
        ax.set_xlabel('${}$'.format(filt.replace('_cor', '').replace('F', 'm_{F').replace('W', 'W}')), fontsize=label_fontsize)
        ax.set_ylabel('${\#}$', fontsize=label_fontsize)
        if narratio_file is not None:
            pass
            # ax = add_narratio_to_plot(ax, target, ratio_data, mid_txt='RGB')

        #plt.tick_params(labelsize=16)
        outfile = '{}_{}{}_lfs{}'.format(lf_file.split('_lf')[0], filt, extra_str, EXT)
        plt.tight_layout()
        ax.grid()
        ax.tick_params(labelsize=ticklabel_fontsize)
        plt.savefig(outfile)
        logger.info('wrote {}'.format(outfile))
    return


def add_trgb(ax, target, band, lf=True, regions_kw=None):
    faint_color = 'k'
    ax = add_lines_to_plot(ax, faint_color=faint_color, lf=lf,
                           **regions_kw)
    return ax


def add_lines_to_plot(ax, mag_bright=None, mag_faint=None, offset=0.,
                      trgb=None, trgb_exclude=0., lf=True, col_min=None,
                      col_max=None, faint_color=None, **kwargs):

    faint_color = faint_color or 'black'

    if mag_faint is not None:
        low = mag_faint
    else:
        low = trgb + offset

    yarr = np.linspace(*ax.get_ylim())

    if lf:
        # vertical lines around the trgb exclude region
        fill_between = ax.fill_betweenx
        axline = ax.axvline
    else:
        fill_between = ax.fill_between
        axline = ax.axhline
        if not None in [col_min, col_max]:
            ax.axvline(col_min, color='black')
            ax.axvline(col_max, color='black')

    if trgb_exclude > 0:
        fill_between(yarr, trgb - trgb_exclude, trgb + trgb_exclude,
                     color='black', alpha=0.1)
    axline(trgb, color='black', linestyle='--')
    if offset > 0 or mag_faint is not None:
        axline(low, color=faint_color, linestyle='--')

    return ax


def diag_cmd(trilegal_catalog, lf_file, regions_kw={}, Av=0.,
             target=None, optfilter1='', use_exclude=False, zcolumns='stage'):
    """
    A two column plot with a data CMD and a scaled model CMD with stages
    pointed out.
    """
    opt_lfd, nir_lfd = fileio.load_lf_file(lf_file)

    sgal = SimGalaxy(trilegal_catalog)
    if 'dav' in trilegal_catalog.lower():
        print('applying dav')
        dAv = float('.'.join(sgal.name.split('dav')[1].split('.')[:2]).replace('_',''))
        sgal.data['F475W'] += sgal.apply_dAv(dAv, 'F475W', 'phat', Av=Av)
        sgal.data['F814W'] += sgal.apply_dAv(dAv, 'F814W', 'phat', Av=Av)
    filter1, filter2 = [f for f in sgal.name.split('_') if f.startswith('F')]
    if type(zcolumns) is str:
        zcolumns = [zcolumns]

    optgal, nirgal = fileio.load_obs(target, optfilter1=optfilter1)

    if opt:
        if 'm31' in trilegal_catalog or 'B' in trilegal_catalog:
            mag1 = 'F475W'
            mag2 = 'F814W'
        else:
            mag1, = [m for m in optgal.data.dtype.names
                     if m.startswith('MAG1') and not 'ERR' in m and not 'STD' in m]
            mag2, = [m for m in optgal.data.dtype.names
                     if m.startswith('MAG2') and not 'ERR' in m and not 'STD' in m]
        gal = optgal
        band = 'opt'
        inds = opt_lfd['optidx_norm'][0]
    else:
        if 'm31' in trilegal_catalog or 'B' in trilegal_catalog:
            mag1 = 'F475W'
            mag2 = 'F814W'
        else:
            mag1 = 'MAG1_IR'
            mag2 = 'MAG2_IR'
            filter1 = 'F110W'
            filter2 = 'F160W'
        gal = nirgal
        band = 'nir'
        inds = nir_lfd['niridx_norm'][0]

    outfmt = trilegal_catalog.replace('.dat', '')
    for zcolumn in zcolumns:
        zstr = zcolumn.translate(None, '/[]')
        ylim = None
        if zcolumn == 'm_ini':
            ylim = (0.9, 8.)
        elif zcolumn == 'stage':
            ylim = (0, 9)
        elif zcolumn == 'logML':
            ylim = (-10.5, -4)
        elif zcolumn == 'logAge':
            ylim = (6, 10.1)

        if filter2 == 'F814W':
            maglim = (28, 22)
            collim = (-2, 5)
        else:
            maglim = (25, 18)
            collim = (-0.5, 2)

        # hist-scatter plot
        magbins = np.arange(16, 27, 0.1)
        axs = sgal.scatter_hist(filter2, zcolumn, coldata='stage', xbins=magbins,
                                ybins=50, slice_inds=inds, ylim=ylim, xlim=maglim)
        dhist = np.histogram(gal.data[mag2], bins=magbins)[0]
        mhist = np.histogram(sgal.data[filter2][inds], bins=magbins)[0]
        axs[0].plot(magbins[1:], dhist, color='k', linestyle='steps', lw=2,
                    zorder=1, alpha=0.3)
        axs[0].plot(magbins[1:], mhist, color='r', linestyle='steps', lw=2,
                    zorder=1, alpha=0.3)

        outfile = '{}_{}_{}_scatterhist{}'.format(outfmt, zstr, band, EXT)
        plt.savefig(outfile)
        logger.info('wrote {}'.format(outfile))

        # data model CMD plot
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True,
                                       figsize=(12, 8))

        ax1.plot(gal.data[mag1] - gal.data[mag2], gal.data[mag2], '.',
                 color='k', alpha=0.2, zorder=1)

        ax2 = sgal.color_by_arg(sgal.data[filter1] - sgal.data[filter2],
                                filter2, zcolumn, ax=ax2, ylim=ylim,
                                slice_inds=inds)
        for ax in [ax1, ax2]:
            if use_exclude:
                if filter2 == 'F814W':
                    match_param = fileio.find_match_param(target, optfilter1=optfilter1)
                    exg = stellar_pops.get_exclude_gates(match_param)
                    # exg are V-I vs V not V-I vs I...
                    ax.plot(exg[:, 0], exg[:, 1] - exg[:, 0], lw=2)
            ax.set_ylim(maglim)
            ax.set_xlim(collim)
            ax.set_ylabel(r'${}$'.format(filter2))
            ax.set_xlabel(r'${}-{}$'.format(filter1, filter2))

            if len(regions_kw) > 0:
                ax = add_lines_to_plot(ax, lf=False, **regions_kw)

        outfile = '{}_{}_{}{}'.format(outfmt, zstr, band, EXT)
        ax.tick_params(labelsize=ticklabel_fontsize)
        plt.tight_layout()
        plt.savefig(outfile)
        logger.info('wrote {}'.format(outfile))
    return


def main(argv=None):
    from ..analysis.normalize import parse_regions
    parser = argparse.ArgumentParser(description="Plot LFs against galaxy data")

    parser.add_argument('-b', '--comp_frac', type=float, default=None,
                        help='completeness fraction')

    parser.add_argument('-c', '--colorlimits', type=str, default=None,
                        help='comma separated color min, color max, opt then nir')

    parser.add_argument('-d', '--diag',  action='store_true',
                        help='trilegal catalog to make a diagnostic cmd instead of plotting LFs')

    parser.add_argument('-e', '--trgbexclude', type=float, default=0.1,
                        help='comma separated regions around trgb to exclude')

    parser.add_argument('-g', '--trgboffset', type=float, default=1.,
                        help='trgb offset, mags below trgb')

    parser.add_argument('-m', '--maglimits', type=str, default=None,
                        help='comma separated faint and bright yaxis mag limits')

    parser.add_argument('-n', '--narratio_file', type=str,
                        help='model narratio file')

    parser.add_argument('-q', '--colnames', type=str, default='MAG2_ACS,MAG4_IR',
                        help='comma separated column names in observation data')

    parser.add_argument('-r', '--table', type=str,
                        help='read colorlimits, completness mags from a prepared table')

    parser.add_argument('-s', '--scolnames', type=str, default='F814W,F160W',
                        help='comma separated column names in trilegal catalog')

    parser.add_argument('-t', '--trgb', type=float, default=None,
                        help='trgb mag')

    parser.add_argument('-v', '--Av', type=float, default=0.,
                        help='visual extinction')

    parser.add_argument('-y', '--yfilter', type=str, default='I',
                        help='V or I filter to use as y axis of CMD [V untested!]')

    parser.add_argument('-z', '--match_param', type=str, default=None,
                        help='exclude gates from calcsfh parameter file')

    parser.add_argument('observation', type=str,
                        help='data file to compare to')

    parser.add_argument('lf_file', type=str,
                        help='model LFs file')

    args = parser.parse_args(argv)

    args.target = args.lf_file.split('_')[0]  # maybe this should be command line?

    regions_kw = parse_regions(args)
    col1, col2 = args.colnames.split(',')

    filter1, filter2 = args.scolnames.split(',')

    if args.diag:
        zcols = ['stage', 'logAge', 'm_ini', '[M/H]', 'C/O', 'logML']
        diag_cmd(args.cmd, args.lf_file[0], regions_kw=optregions_kw,
                 use_exclude=args.use_exclude, zcolumns=zcols)
        diag_cmd(args.cmd, args.lf_file, opt=False, regions_kw=nirregions_kw,
                 use_exclude=args.use_exclude, zcolumns=zcols, Av=args.Av)
    else:
        if args.narratio_file is None:
            narratio_file = args.lf_file.replace('lf', 'narratio')
            if os.path.isfile(narratio_file):
                args.narratio_file = narratio_file
        agb_mod = args.lf_file.split('_')[3]
        compare_to_gal(args.lf_file, args.observation,
                       narratio_file=args.narratio_file, filter1=filter1,
                       agb_mod=agb_mod, regions_kw=regions_kw,
                       xlims=[[19,28], [18, 25]], filter2=filter2,
                       col1=col1, col2=col2, match_param=args.match_param)

if __name__ == '__main__':
    sys.exit(main())
