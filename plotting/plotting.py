import argparse
import difflib
import logging
import os
import sys

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import ResolvedStellarPops as rsp
angst_data = rsp.angst_tables.angst_data

from ..analysis.analyze import get_itpagb
from ..fileio import load_obs, find_fakes, find_match_param, load_lf_file
from ..fileio import load_observation
from ..utils import minmax

from ..pop_synth import stellar_pops
from ..sfhs import star_formation_histories
# where the matchfake files live
from ..TPAGBparams import snap_src
matchfake_loc = os.path.join(snap_src, 'data', 'galaxies')
data_loc = os.path.join(snap_src, 'data', 'opt_ir_matched_v2')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def add_narratio_to_plot(ax, target, ratio_data, mid_txt='RGB'):
    stext_kw = dict({'color': 'black', 'fontsize': 14, 'ha': 'center'}.items() +
                    rsp.graphics.GraphicsUtils.ann_kwargs.items())
    dtext_kw = dict(stext_kw.items() + {'color': 'darkred'}.items())

    assert ratio_data[0]['target'] == 'data', \
        'the first line of the narratio file needs to be a data measurement'
    nagb = float(ratio_data[0]['nagb'])
    nrgb = float(ratio_data[0]['nrgb'])

    dratio = nagb / nrgb
    dratio_err = rsp.utils.count_uncert_ratio(nagb, nrgb)

    indx, = np.nonzero(ratio_data['target'] == target)
    mrgb = float(ratio_data[indx]['nrgb'])
    magb = float(ratio_data[indx]['nagb'])

    #yval = 1.2  # text yloc found by eye, depends on fontsize
    stext_kw['transform'] = ax.transAxes
    dtext_kw['transform'] = ax.transAxes
    yval = 0.95
    xagb_val = 0.17
    xrgb_val = 0.5
    xratio_val = 0.83
    xvals = [xagb_val, xrgb_val, xratio_val]

    # simulated nrgb and nagb are the mean values
    srgb_text = r'$\langle N_{\rm %s}\rangle =%i$' % (mid_txt, np.mean(mrgb))
    sagb_text = r'$\langle N_{\rm TP-AGB}\rangle=%i$' % np.mean(magb)

    # one could argue taking the mean isn't the best idea for
    # the ratio errors.
    sratio_text = '$f=%.3f\pm%.3f$' % (np.mean(ratio_data['ar_ratio']),
                                       np.mean(ratio_data['ar_ratio_err']))

    drgb_text = r'$N_{\rm %s}=%i$' % (mid_txt, nrgb)
    dagb_text = r'$N_{\rm TP-AGB}=%i$' % nagb
    dratio_text =  '$f = %.3f\pm%.3f$' % (dratio, dratio_err)

    textss = [[sagb_text, srgb_text, sratio_text],
             [dagb_text, drgb_text, dratio_text]]
    kws = [stext_kw, dtext_kw]

    for kw, texts in zip(kws, textss):
        for xval, text in zip(xvals, texts):
            ax.text(xval, yval, text, **kw)
        yval -= .05  # stack the text
    return ax


def plot_model(mag2s=None, bins=None, norms=None, inorm=None, ax=None,
               plt_kw={}, maglimit=None, agb_mod=None):
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
    
    plt_kw = dict({'linestyle': 'steps-mid', 'color': 'black',
                   'alpha': 0.2}.items() + plt_kw.items())
    
    if agb_mod is not None:
        label = r'$%s$' % agb_mod.split('_')[-1]
        plt_kw_lab = dict(plt_kw.items() + {'label': label}.items())
    else:
        plt_kw_lab = plt_kw
    
    if ax is None:
        fig, (ax) = plt.subplots(figsize=(12, 6))
        
    for i in range(len(mag2s)):
        if inorm is not None:
            #import pdb; pdb.set_trace()
            mag2 = mag2s[i][inorm[i]]
            norm = 1.
        else:
            mag2 = mag2s[i]
            norm = norms[i]

        if maglimit is not None:
            inds, = np.nonzero(mag2 <= maglimit)
        else:
            inds = np.arange(len(mag2))

        hist = np.histogram(mag2[inds], bins=bins)[0]
        # only write legend once
        if i != 0:
            kw = plt_kw
        else:
            kw = plt_kw_lab

        ax.plot(bins[1:], hist * norm, **kw)
    return ax


def plot_gal(mag2, bins, ax=None, target=None, plot_kw={}, fake_file=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    plot_kw = dict({'drawstyle': 'steps-mid', 'color': 'darkred', 'lw': 1,
                     'label': '$%s$' % target}.items() + plot_kw.items())

    hist = np.histogram(mag2, bins=bins)[0]
    err = np.sqrt(hist)
    ax.errorbar(bins[1:], hist, yerr=err, **plot_kw)

    if fake_file is not None:
        comp_corr = stellar_pops.completeness_corrections(fake_file, bins)

    hist = np.histogram(mag2, bins=bins)[0]
    hist /= comp_corr[1:]
    err = np.sqrt(hist)
    plot_kw['lw'] += 1
    ax.errorbar(bins[1:], hist, yerr=err, **plot_kw)

    return ax


def plot_models(lf_file, bins, filt, maglimit=None, ax=None, plt_kw=None,
                agb_mod=None):
    plt_kw = plt_kw or {}
    lfd = load_lf_file(lf_file)
    
    mags = lfd[filt]
        
    ax = plot_model(mag2s=mags, bins=bins, inorm=lfd['idx_norm'],
                    maglimit=maglimit, ax=ax, plt_kw=plt_kw, agb_mod=agb_mod)

    plt_kw['color'] = 'darkgreen'
    ax = plot_model(mag2s=mags, bins=bins, inorm=lfd['sim_agb'],
                    maglimit=maglimit, ax=ax, plt_kw=plt_kw, agb_mod='TP-AGB')

    return ax


def mag2Mag(mag2, target, filter2):
    angst_target = \
    difflib.get_close_matches(target.upper(),
                              angst_data.targets)[0].replace('-', '_')

    if 'F160W' in filter2:
        _, av, dmod = angst_data.get_snap_trgb_av_dmod(angst_target)
    if 'F814W' in filter2:
        target_row = angst_data.__getattribute__(angst_target)
        try:
            key, = [k for k in target_row.keys() if ',' in k]
        except:
            #print target_row
            key = [k for k in target_row.keys() if ',' in k][0]
        av = target_row[key]['Av']
        dmod = target_row[key]['dmod']

    mag = rsp.astronomy_utils.mag2Mag(mag2, filter2, 'wfc3snap', Av=av,
                                      dmod=dmod)
    
    return mag


def compare_lfs(lf_files, filter1='F814W_cor', filter2='F160W_cor',
                col1='MAG2_ACS', col2='MAG4_IR', dmag=0.1, extra_str=''):
    """
    3 panel plot of LF, (data-model)/model, data-model
    doesn't work on lf files with many runs...
    """
    # for opt, and nir...
    # hist data, with completeness corrections
    # do something around trgb, 90% completeness, norm region?
    # hist model -- if more than one, average/mediad
    # model - data / data
    # model - data
    # plot all three
    galaxies = rsp.fileio.get_files(data_loc, '*fits')
    
    fig1, opt_axs = plt.subplots(nrows=3, sharex=True, figsize=(6, 9))
    fig2, nir_axs = plt.subplots(nrows=3, sharex=True, figsize=(6, 9))

    bins = np.arange(16, 27, dmag)
    
    for i, lf_file in enumerate(lf_files):
        lfd = load_lf_file(lf_file)
        target = os.path.split(lf_file)[1].split('_')[0]
        observation, = [g for g in galaxies if target.upper() in g]
        mag1, mag2 = load_observation(observation, col1, col2)
        color = mag1 - mag2
        
        search_str = '*{}*.matchfake'.format(target.upper())    
        fakes = rsp.fileio.get_files(matchfake_loc, search_str)
        nirfake, = [f for f in fakes if 'IR' in f]
        optfake = [f for f in fakes if not 'IR' in f][0]

        for i, filt in enumerate([filter1, filter2]):
            if i == 0:
                axs = opt_axs
                mag = mag1
                fake_file = optfake
            else:
                axs = nir_axs
                mag = mag2
                fake_file = nirfake

            Mbins = mag2Mag(bins, target, filt.replace('_cor', ''))
            #ax.errorbar(bins[1:], hist, yerr=err, **plot_kw)
    
            
            comp_corr = stellar_pops.completeness_corrections(fake_file,
                                                              bins)
            data = np.array(np.histogram(mag2, bins=bins)[0], dtype=float)
            # mask 0s or they will be turned to Abs Mag
            data[data == 0] = np.nan
            data /= comp_corr[1:]

            data = mag2Mag(data, target, filt.replace('_cor', ''))
            err = np.sqrt(data)
            
            smag = np.concatenate(lfd[filt])
            inorm = np.concatenate(lfd['idx_norm'])
            smag_scaled = smag[inorm]
                        
            model = np.array(np.histogram(smag_scaled, bins=bins)[0],
                             dtype=float)
            # mask 0s or they will be turned to Abs Mag
            model[model == 0] = np.nan
            
            model = mag2Mag(model, target, filt.replace('_cor', ''))
            
            dmdiff = data - model
            sdiff = dmdiff / data
            
            axs[0].errorbar(Mbins[1:], model, yerr=np.sqrt(err),
                            linestyle='steps-mid')
            axs[1].plot(Mbins[1:], dmdiff, drawstyle='steps-mid')
            axs[2].plot(Mbins[1:], sdiff, drawstyle='steps-mid')

    for axs in [opt_axs, nir_axs]:
        axs[0].set_yscale('log')
        axs[0].set_ylabel('$\#$')
        axs[1].set_ylabel(r'($N_{data} - N_{model}) / N_{data}$')
        axs[2].set_ylabel(r'$N_{data} - N_{model}$')

    opt_axs[2].set_xlabel(r'${}$'.format(filter1.replace('_cor', '')))
    nir_axs[2].set_xlabel(r'${}$'.format(filter2.replace('_cor', '')))

    [opt_axs[i].set_xlim(-10, 3) for i in range(3)]
    [nir_axs[i].set_xlim(-10, -1) for i in range(3)]
    [opt_axs[i].set_ylim(-500, 500) for i in [1,2]]
    [nir_axs[i].set_ylim(-500, 500) for i in [1,2]]
    fig1.savefig('{}_{}_comp_lfs.png'.format(extra_str, filter1))
    fig2.savefig('{}_{}_comp_lfs.png'.format(extra_str, filter2))
    return opt_axs, nir_axs


def load_data(opt=True, optfilter1=None, target=None, extra_str='',
              optfilter2_limit=None, nirfilter2_limit=None, 
              optregions_kw={}, nirregions_kw={}):
              
    optgal, nirgal = load_obs(target, optfilter1=optfilter1)
    optfake, nirfake = find_fakes(target)
    if opt:
        maglimit = optfilter2_limit
        fake_file = optfake
        extra_str += '_opt'
        regions_kw = optregions_kw
        try:   
            mag2 = optgal.data['MAG2_ACS']
        except:
            try:
                mag2 = optgal.data['MAG2_WFPC2']
            except:
                mag2 = optgal.data['F814W']
        filter2 = optfilter2
    else:
        maglimit = nirfilter2_limit
        fake_file = nirfake
        try:
            mag2 = nirgal.data['MAG2_IR']
        except:
            mag2 = optgal.data['F814W']
        extra_str = extra_str.replace('opt', 'nir')
        regions_kw = nirregions_kw
        filter2 = nirfilter2
    
    return mag2, filter2, regions_kw, fake_file, maglimit, extra_str


def compare_to_gal(lf_file, observation, filter1='F814W_cor',
                   filter2='F160W_cor', col1='MAG2_ACS', col2='MAG4_IR',
                   dmag=0.1, narratio_file=None, make_plot=True,
                   regions_kw=None, xlim=None, ylim=None, extra_str='',
                   agb_mod=None, mplt_kw={}, dplot_kw={}):
    '''
    Plot the LFs and galaxy LF.

    ARGS:
    narratio: overlay NRGB, NAGB, and NAGB/NRGB +/- err
    no_agb: plot the LF without AGB stars

    RETURNS:
    ax1, ax2: axes instances created for the plot.

    '''
    target = os.path.split(lf_file)[1].split('_')[0]

    mag1, mag2 = load_observation(observation, col1, col2)
    #color = mag1 - mag2
    #ogal, ngal = load_obs(target)
    #mag1 = ogal.data['MAG2_ACS']
    #mag2 = ngal.data['MAG2_IR']
    
    search_str = '*{}*.matchfake'.format(target.upper())    
    fakes = rsp.fileio.get_files(matchfake_loc, search_str)
    nirfake, = [f for f in fakes if 'IR' in f]
    optfake = [f for f in fakes if not 'IR' in f][0]
    
    if narratio_file is not None:
        ratio_data = rsp.fileio.readfile(narratio_file, string_column=[0, 1, 2])

    if 'cor' in filter1:
        if not '_ast_cor' in extra_str:
            extra_str += '_ast_cor'

    for filt, fake_file in zip([filter1, filter2], [optfake, nirfake]):
        mag = mag2 
        if filt == filter1:
            mag = mag1
        bins = np.arange(mag.min(), mag.max(), step=dmag)
        ax = plot_models(lf_file, bins, filt, plt_kw=mplt_kw, agb_mod=agb_mod)

        # plot galaxy data
        ax = plot_gal(mag, bins, ax=ax, target=target, fake_file=fake_file,
                      plot_kw=dplot_kw)

        ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(1, ax.get_ylim()[-1])
    
        if xlim is not None:
            ax.set_xlim(xlim)
        #else:
        #    xmax = ax.get_xlim()[-1]
        #    xmin = np.min([np.min(mag2), np.min(np.concatenate(mag2s))])
        #    ax.set_xlim(xmin, xmax)

        if regions_kw is not None:
            ax = add_lines_to_plot(ax, **regions_kw)
        
        ax.legend(loc='lower right')
        ax.set_xlabel('${}$'.format(filt.replace('_cor', '')), fontsize=20)

        if narratio_file is not None:
            ax = add_narratio_to_plot(ax, target, ratio_data, mid_txt='RGB')

        plt.tick_params(labelsize=16)
        outfile = '{}_{}{}_lfs.png'.format(lf_file.split('_lf')[0], filt, extra_str)
        plt.savefig(outfile)
        print 'wrote {}'.format(outfile)
    return


def add_lines_to_plot(ax, mag_bright=None, mag_faint=None, offset=None,
                      trgb=None, trgb_exclude=None, lf=True, col_min=None,
                      col_max=None, **kwargs):
    
    if mag_bright is not None:
        low = mag_faint
        mid = mag_bright
    else:
        assert offset is not None, \
        'need either offset or mag limits'
        low = trgb + offset
        mid = trgb + trgb_exclude
    
    yarr = np.linspace(*ax.get_ylim())
    if lf:
        
        # vertical lines around the trgb exclude region
        ax.fill_betweenx(yarr, trgb - trgb_exclude, trgb + trgb_exclude,
                         color='black', alpha=0.1)
        ax.vlines(trgb, *ax.get_ylim(), color='black', linestyle='--')
        
        ax.vlines(low, *ax.get_ylim(), color='black', linestyle='--')
        #ax.fill_betweenx(yarr, mid, low, color='black', alpha=0.1)
        #if mag_limit_val is not None:
        #    ax.fill_betweenx(yarr, mag_limit_val, ax.get_xlim()[-1],
        #                     color='black', alpha=0.5)
    else:
        xarr = np.linspace(*ax.get_xlim())
        
        # vertical lines around the trgb exclude region
        ax.fill_between(yarr, trgb - trgb_exclude, trgb + trgb_exclude,
                         color='black', alpha=0.1)
        ax.hlines(trgb, *ax.get_xlim(), color='black', linestyle='--')
        
        ax.hlines(low, *ax.get_xlim(), color='black', linestyle='--')
        #ax.fill_betweenx(yarr, mid, low, color='black', alpha=0.1)
        #if mag_limit_val is not None:
        #    ax.fill_betweenx(yarr, mag_limit_val, ax.get_xlim()[-1],
        #                     color='black', alpha=0.5)
        if not None in [col_min, col_max]:
            ax.vlines(col_min, *ax.get_ylim(), color='black')
            ax.vlines(col_max, *ax.get_ylim(), color='black')
    return ax


def diag_cmd(trilegal_catalog, lf_file, regions_kw={}, Av=0.,
             target=None, optfilter1='', use_exclude=False, zcolumns='stage'):
    """
    A two column plot with a data CMD and a scaled model CMD with stages
    pointed out.
    """
    opt_lfd, nir_lfd = load_lf_file(lf_file)

    sgal = rsp.SimGalaxy(trilegal_catalog)
    if 'dav' in trilegal_catalog.lower():
        print('applying dav')
        dAv = float('.'.join(sgal.name.split('dav')[1].split('.')[:2]).replace('_',''))
        sgal.data['F475W'] += sgal.apply_dAv(dAv, 'F475W', 'phat', Av=Av)
        sgal.data['F814W'] += sgal.apply_dAv(dAv, 'F814W', 'phat', Av=Av)
    filter1, filter2 = [f for f in sgal.name.split('_') if f.startswith('F')]
    #import pdb; pdb.set_trace()
    if type(zcolumns) is str:
        zcolumns = [zcolumns]
    
    optgal, nirgal = load_obs(target, optfilter1=optfilter1)
    
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
        axs[0].plot(magbins[1:], dhist, color='k', linestyle='steps-pre', lw=2,
                    zorder=1, alpha=0.3)
        axs[0].plot(magbins[1:], mhist, color='r', linestyle='steps-pre', lw=2,
                    zorder=1, alpha=0.3)

        outfile = '{}_{}_{}_scatterhist.png'.format(outfmt, zstr, band)
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
                    match_param = find_match_param(target, optfilter1=optfilter1)
                    exg = stellar_pops.get_exclude_gates(match_param)
                    # exg are V-I vs V not V-I vs I...
                    ax.plot(exg[:, 0], exg[:, 1] - exg[:, 0], lw=2)
            ax.set_ylim(maglim)
            ax.set_xlim(collim)
            ax.set_ylabel(r'${}$'.format(filter2))
            ax.set_xlabel(r'${}-{}$'.format(filter1, filter2))
            
            if len(regions_kw) > 0:
                ax = add_lines_to_plot(ax, lf=False, **regions_kw)
        
        outfile = '{}_{}_{}.png'.format(outfmt, zstr, band)
        plt.savefig(outfile)
        logger.info('wrote {}'.format(outfile))
    return


def main(argv):
    from ..analysis.normalize import parse_regions
    parser = argparse.ArgumentParser(description="Plot LFs against galaxy data")
    
    parser.add_argument('-c', '--colorlimits', type=str, default=None,
                        help='comma separated color min, color max, opt then nir')

    parser.add_argument('-e', '--trgbexclude', type=str, default='0.1,0.2',
                        help='comma separated regions around trgb to exclude')

    parser.add_argument('-f', '--optfilter1', type=str,
                        help='optical V filter')

    parser.add_argument('-m', '--maglimits', type=str, default=None,
                        help='comma separated mag faint, mag bright, opt then nir')

    parser.add_argument('-o', '--trgboffsets', type=str, default=None,
                        help='comma separated trgb offsets')
    
    parser.add_argument('-r', '--table', type=str,
                        help='read colorlimits, completness mags from a prepared table')
    
    parser.add_argument('-u', '--use_exclude', action='store_true',
                        help='decontaminate LF by excluding stars within exclude_gates')
    
    parser.add_argument('-n', '--narratio_file', type=str,
                        help='model narratio file')

    parser.add_argument('-d', '--cmd', type=str,
                        help='trilegal catalog to make a diagnostic cmd instead of plotting LFs')

    parser.add_argument('-v', '--Av', type=float, default=0.,
                        help='visual extinction')
    
    parser.add_argument('lf_file', type=str, nargs='*',
                        help='model LFs file')

    parser.add_argument('observation', type=str,
                        help='data file to compare to')
        
    parser.add_argument('agb_mod', type=str,
                        help='agb model name')

    args = parser.parse_args(argv)

    #optregions_kw, nirregions_kw = parse_regions(args)
    

    if args.cmd:
        zcols = ['stage', 'logAge', 'm_ini', '[M/H]', 'C/O', 'logML']
        diag_cmd(args.cmd, args.lf_file[0], regions_kw=optregions_kw,
                 use_exclude=args.use_exclude, zcolumns=zcols)
        diag_cmd(args.cmd, args.lf_file, opt=False, regions_kw=nirregions_kw,
                 use_exclude=args.use_exclude, zcolumns=zcols, Av=args.Av)
    elif len(args.lf_file) > 1:
        compare_lfs(args.lf_file, extra_str=args.agb_mod)
    else:
        compare_to_gal(args.lf_file[0], args.observation,
                       narratio_file=args.narratio_file,
                       agb_mod=args.agb_mod, xlim=None, ylim=None)

if __name__ == '__main__':
    main(sys.argv[1:])
    
# everything below is old... might not need ... probably broken


def plot_random_sfhs(vsfh):
    '''
    plot the random sfr arrays with the data sfr arrays.
    Hard coded file locations. So read the code.
    '''
    sfr_file_loc = vsfh.outfile_loc
    hmc_file_loc = vsfh.base

    outfile = os.path.join(sfr_file_loc, '%s_random_sfr.png' % vsfh.target)
    hmc_file, = rsp.fileIO.get_files(hmc_file_loc, '%s*mcmc.zc' % vsfh.target)

    sfh = star_formation_histories(sfh_file, hmc_file=hmc_file,
                                   sfr_file_loc=sfr_file_loc,
                                   sfr_file_search_fmt='*sfr')

    sfh.plot_sfh('sfr', plot_random_arrays_kw={'from_files': True},
                 outfile=outfile)
    return


def tpagb_masses(chi2file, band='opt', model_src='default', dry_run=False,
                 mass=True, old=False):
    '''
    using the chi2file run trilegal with the best fit sfh and return the
    normalization and tp-agb masses (scaled simulation)
    '''
    if model_src == 'default':
        model_src = snap_src + '/models/varysfh/'

    components = os.path.split(chi2file)[1].split('_')
    model = '_'.join(components[:3])
    target = components[3]
    model_loc = os.path.join(model_src, target, model, 'mc')

    # read chi2 file
    chi2_data = rsp.fileIO.readfile(chi2file)

    # best fitting chi2 run in band
    ibest_fit = np.argmin(chi2_data['%s_chi2' % band])

    # should work out to isfr == ibest_fit, but just in case:
    isfr = chi2_data['sfr'][ibest_fit]

    # associated input file for best run
    tri_inp, = rsp.fileIO.get_files(model_loc, '*%03d.dat' % isfr)

    # run trilegal with best run
    tri_outp = tri_inp.replace('.dat', '_%s_best.dat' % band).replace('inp', 'outp')
    cmd_input = 'cmd_input_%s.dat' % model.upper()
    rsp.TrilegalUtils.run_trilegal(cmd_input, tri_inp, tri_outp,
                                   dry_run=dry_run)

    # load trilegal run and do the normalization (I should really save that...)
    # all that's needed here is opt_norm and ir_norm.
    filter1 = sfh_tests.get_filter1(target)
    ags=sfh_tests.load_default_ancient_galaxies()

    files = sfh_tests.FileIO()
    files.read_trilegal_catalog(tri_outp, filter1=filter1)
    files.load_data_for_normalization(target=target, ags=ags)
    files.load_trilegal_data()
    sopt_rgb, sir_rgb, sopt_agb, sir_agb = \
        sfh_tests.rgb_agb_regions(files.sgal, files.opt_offset,
                                             files.opt_trgb, files.opt_trgb_err,
                                             ags, files.ir_offset,
                                             files.ir_trgb, files.ir_trgb_err,
                                             files.opt_mag, files.ir_mag)
    opt_norm, ir_norm, opt_rgb, ir_rgb, opt_agb, ir_agb = \
        sfh_tests.normalize_simulation(files.opt_mag, files.ir_mag,
                                                  files.nopt_rgb, files.nir_rgb,
                                                  sopt_rgb, sir_rgb, sopt_agb,
                                                  sir_agb)
    print target, opt_norm, ir_norm
    with open(tri_inp.replace('.dat', '_norms.dat'), 'w') as out:
        out.write('# opt_norm ir_norm\n')
        out.write('%.6f %.6f \n' % (opt_norm, ir_norm))

    if band == 'opt':
        norm = opt_norm
    if band == 'ir':
        norm = ir_norm

    # load tp-agb stars
    files.sgal.all_stages('TPAGB')
    # it's crazy to believe there is a tp-agb star bluer than the color cut
    # but to be consistent with the rest of the method, will do a color cut.
    cut_inds = files.__getattribute__('%s_color_cut' % band)
    itpagb = list(set(files.sgal.itpagb) & set(cut_inds))

    # grab TP-AGB masses

    if mass is True:
        mass = files.sgal.data.get_col('m_ini')
        ret_val = mass[itpagb]
    else:
        met = files.sgal.data.get_col('[M/H]')
        if old is True:
            olds, = np.nonzero(files.sgal.data.get_col('logAge') > 8.5)
            itpagb = list(set(files.sgal.itpagb) & set(cut_inds) & set(olds))
        ret_val = met[itpagb]
    return ret_val, norm


def model_cmd_withasts(fname=None, sgal=None, filter1=None, filter2=None,
                       trgb=None, trgb_exclude=None, mag_faint=None, inorm=None,
                       xlim=(-.5, 5), mag_bright=None, mag_limit_val=None,
                       agb=None, rgb=None, **kwargs):
    """plot cmd and ast corrected cmd side by side"""
    if sgal is None:
        sgal = rsp.SimGalaxy(fname, filter1=filter1, filter2=filter2)
    sgal.load_ast_corrections()

    fig, axs = plt.subplots(ncols=2, figsize=(12,8), sharex=True, sharey=True)
    if inorm is None:
        axs[0] = sgal.plot_cmd(sgal.color, sgal.mag2, ax=axs[0])
        inorm = np.arange(len(sgal.ast_color))
    else:
        axs[1] = sgal.plot_cmd(sgal.ast_color, sgal.ast_mag2, ax=axs[1])

    ylim = axs[0].get_ylim()

    axs[1] = sgal.plot_cmd(sgal.ast_color[inorm], sgal.ast_mag2[inorm],\
                           ax=axs[1])
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    [ax.set_xlabel(r'$%s-%s$' % (filter1, filter2), fontsize=20) for ax in axs]
    axs[0].set_ylabel(r'$%s$' % filter2, fontsize=20)
    axs[1].set_title(r'$\emph{%s}$' % sgal.name.replace('_', r'\_'),
                     fontsize=12)

    ax = axs[1]
    xarr = np.linspace(*ax.get_xlim())
        # vertical lines around the trgb exclude region
    ax.fill_between(xarr, trgb - trgb_exclude, trgb + trgb_exclude,
                    color='black', alpha=0.1)
    ax.hlines(trgb, *ax.get_xlim(), color='black', linestyle='--')
    if not None in [mag_faint, mag_bright]:
        ax.fill_between(xarr, mag_faint, mag_bright, color='black', alpha=0.1)
    if mag_limit_val is not None:
        ax.fill_between(xarr, mag_limit_val, ax.get_ylim()[0], color='black',
                        alpha=0.1)
    if agb is not None:
        ax.plot(sgal.ast_color[agb], sgal.ast_mag2[agb], 'o', color='red',
                mec='none', alpha=0.3)
    if rgb is not None:
        ax.plot(sgal.ast_color[rgb], sgal.ast_mag2[rgb], '.', color='darkred',
                mec='none', alpha=0.3)

    plt.savefig(os.path.join(sgal.base, sgal.name).replace('.dat', 'cmds.png'))
    plt.close()
    return #axs, sgal


def tpagb_mass_histograms(chi2_location='draft_run', band='opt', dry_run=True,
                         model='nov13', model_src='default'):
    '''
    plot a histogram of the scaled number of tpagb stars for the best fitting
    model in each model chi2 file in the chi2_location

    all args besides chi2_location are passed to tpagb_masses.

    the trilegal output files can be in a difference location with directories
    model_src/target/model/mc see tpagb_masses.

    plot colors are fixed at 6: errors for more than 6 targets.
    '''
    if chi2_location == 'draft_run':
        chi2_location = snap_src + '/models/varysfh/match-hmc/'

    chi2files = rsp.fileIO.get_files(chi2_location, '*%s_*chi2.dat' % model)
    # I do gaussian chi2 too, not just poisson...
    chi2files = [c for c in chi2files if not 'gauss' in c][::-1]

    # get the tpagb masses
    (masses, norm) = zip(*[tpagb_masses(c, band=band, dry_run=dry_run,
                                        model_src=model_src)
                    for c in chi2files])
    norm = np.array(norm)

    ts = [os.path.split(c)[1].split('_')[3] for c in chi2files]
    targets = galaxy_tests.ancients()
    tinds = [ts.index(t.lower()) for t in targets]
    targets = np.array(ts)[tinds]
    labels = ['$%s$' % t.upper().replace('-DEEP', '').replace('-', '\!-\!')
              for t in targets]


    # get the hist made nice and stacked, and then scale it down.
    hists, bins, pps = plt.hist(masses, stacked=True, align='left',
                                histtype='step', bins=50, visible=False)
    plt.close()

    # scaled histograms
    norm_hists = [hists[i] * norm[i] for i in range(len(hists))]

    # actual plot... norm scales the histogram.
    # set up plot
    cols = color_scheme

    fig, ax = plt.subplots()

    # mask 0 values so there is a vertical line on the plot
    for i in range(len(hists)):
        norm_hists[i][norm_hists[i]==0] = 1e-5
        yplot = np.cumsum(norm_hists[i]) / np.sum(norm_hists[i])
        #yplot = norm_hists[i]
        ax.plot(bins[:-1], yplot, linestyle='steps-pre', color='grey', lw=4)
        ax.plot(bins[:-1], yplot, linestyle='steps-pre', color=cols[i],
                lw=2, label=labels[i], alpha=.9)

    #ax.plot(bins[:-1], np.sum(norm_hists, axis=0), linestyle='steps-pre',
    #             color='darkgrey', lw=3, label=r'$\rm{Total}$')
    ax.legend(loc=0, frameon=False)

    #ax.set_yscale('log')
    #ax.set_ylim(3, 10**3)
    ax.set_xlim(0.6, 3)
    ax.set_xlabel(r'$\rm{Mass\ M_\odot}$', fontsize=20)
    ax.set_ylabel(r'$\rm{Cumulative\ Fraction\ of\ {TP\!-\!AGB}\ Stars}$', fontsize=20)
    plt.tick_params(labelsize=16)
    plt.savefig('tpagb_mass_hist_%s_%s.png' % (band, model), dpi=150)
    return ax


def trilegal_metals(chi2_location='draft_run', band='opt', dry_run=False,
                    model='nov13', model_src='default', old=False, feh=False):
    if chi2_location == 'draft_run':
        chi2_location = snap_src + '/models/varysfh/match-hmc/'

    chi2files = rsp.fileIO.get_files(chi2_location, '*%s_*chi2.dat' % model)
    # I do gaussian chi2 too, not just poisson...
    chi2files = [c for c in chi2files if not 'gauss' in c][::-1]

    # get the tpagb masses
    (mhs, norm) = zip(*[tpagb_masses(c, band=band, dry_run=dry_run,
                                        model_src=model_src, mass=False,
                                        old=old) for c in chi2files])
    ts = [os.path.split(c)[1].split('_')[3] for c in chi2files]
    targets = galaxy_tests.ancients()
    tinds = [ts.index(t.lower()) for t in targets]
    targets = np.array(ts)[tinds]
    from ResolvedStellarPops.convertz import convertz
    if feh is True:
        ind = 4
    else:
        ind = 1
    zs = np.array([convertz(mh=i)[ind] for i in mhs])
    for i, target in enumerate(targets):
        print '%.4f %.4f %.4f %s ' % (np.min(zs[i]), np.median(zs[i]),
                                      np.max(zs[i]), target)


   
# below, I had made a way to use lf and scale on the fly not sure if I still want it.

def norm_lf_file(opt_hists, opt_binss, ir_hists, ir_binss, opt_trgb, ir_trgb,
                 trgb_excludes, opt_limit, ir_limit, opt_mag2, ir_mag2):

    def count_stars_from_hist(hist, bins, trgb, trgb_exclude, faint):
        irgb = rsp.utils.between(bins[1:], faint, trgb + trgb_exclude)
        nrgb = np.sum(hist[irgb])
        return float(nrgb)

    opt_gal_hist, opt_gal_bins = np.histogram(opt_mag2, bins=opt_binss[0])
    ir_gal_hist, ir_gal_bins = np.histogram(ir_mag2, bins=ir_binss[0])

    nopt_rgbs = np.array([count_stars_from_hist(opt_hists[i], opt_binss[i],
                                                opt_trgb, trgb_excludes[0],
                                                opt_limit)
                          for i in range(len(opt_hists))])

    nir_rgbs = np.array([count_stars_from_hist(ir_hists[i], ir_binss[i],
                                               ir_trgb, trgb_excludes[1],
                                               ir_limit)
                         for i in range(len(ir_hists))])

    nopt_rgb = count_stars_from_hist(opt_gal_hist, opt_gal_bins, opt_trgb,
                                     trgb_excludes[0], opt_limit)

    nir_rgb = count_stars_from_hist(ir_gal_hist, ir_gal_bins, ir_trgb,
                                    trgb_excludes[1], ir_limit)

    opt_norms = nopt_rgb /nopt_rgbs
    ir_norms = nir_rgb / nir_rgbs
    return opt_norms, ir_norms


def plot_lf_file(opt_lf_file, ir_lf_file, axs=None, plt_kw=None,
                 opt_limit=None, ir_limit=None, agb_mod=None,
                 opt_norms=None, ir_norms=None, norm_lf_kw=None):
    '''needs work, but: plot the lf files.
    norm_lf_kw: opt_trgb, ir_trgb, trgb_excludes, opt_limit, ir_limit,
                opt_mag2, ir_mag2
    '''
    # set up the plot
    norm_lf_kw = norm_lf_kw or {}
    plt_kw = plt_kw or {}
    plt_kw = dict({'linestyle': 'steps-mid', 'color': 'black',
                   'alpha': 0.2}.items() + plt_kw.items())
    if agb_mod is not None:
        label = '%s' % agb_mod.split('_')[-1]
        plt_kw_lab = dict(plt_kw.items() + {'label': label, 'lw': 2}.items())
    if 'label' in plt_kw.keys():
        plt_kw_lab = dict(plt_kw.items() + {'lw': 2, 'alpha': 1}.items())
        del plt_kw['label']

    if axs is None:
        _, (axs) = plt.subplots(ncols=2, figsize=(12, 6))
        plt.subplots_adjust(right=0.95, left=0.05, wspace=0.1)

    opt_hists, opt_binss = load_lf_file(opt_lf_file)
    ir_hists, ir_binss = load_lf_file(ir_lf_file)

    if None in [opt_norms, ir_norms]:
        opt_norms, ir_norms = norm_lf_file(opt_hists, opt_binss, ir_hists,
                                           ir_binss, **norm_lf_kw)

    for i, (hists, binss, limit, norms) in \
        enumerate(zip([opt_hists, ir_hists], [opt_binss, ir_binss],
                      [opt_limit, ir_limit], [opt_norms, ir_norms])):
        for j, (hist, bins, norm) in enumerate(zip(hists, binss, norms)):
            if j == 0:
                axs[i].plot(np.zeros(20)-99, np.zeros(20)-99, **plt_kw_lab)
            if limit is not None:
                inds, = np.nonzero(bins <= limit)
                axs[i].plot(bins[inds], hist[inds] * norm, **plt_kw)
            else:
                axs[i].plot(bins, hist * norm, **plt_kw)

    return axs, opt_binss[0], ir_binss[0]


def ast_corrections_plot(mag1, mag2, mag1_cor, mag2_cor, ymag='I'):
    fig, ax = plt.subplots(figsize=(8, 8))

    rec, = np.nonzero((np.abs(mag1_cor) < 30) & (np.abs(mag2_cor) < 30))
    mag = mag1[rec]
    mag_cor = mag1_cor[rec]
    if ymag == 'I':
        mag = mag2[rec]
        mag_cor = mag2_cor[rec]

    color = mag1[rec] - mag2[rec]
    color_cor = mag1_cor[rec] - mag2_cor[rec]
    dcol = color_cor - color
    dmag = mag_cor - mag
    for i in range(len(rec)):
        if dcol[i] == 0 or dmag[i] == 0:
            continue
        ax.arrow(color[i], mag[i], dcol[i], dmag[i], length_includes_head=True,
                 width=1e-4, color='k', alpha=0.3)


class Plotting(object):
    def __init__(self, input_file=None, kwargs={}):
        self.input_file = input_file
        if input_file is not None:
            kwargs.update(rsp.fileio.load_input(input_file))

        [self.__setattr__(k, v) for k, v in kwargs.items()]


    def count_stars_from_hist(self, opt_hist, opt_bins, ir_hist, ir_bins):
        ratio_data = {}
        for i, (hist, bins, band) in enumerate(zip([opt_hist, ir_hist],
                                                   [opt_bins, ir_bins],
                                                   ['opt', 'ir'])):
            trgb = self.__getattribute__('%s_trgb' % band)
            trgb_err = self.__getattribute__('%s_trgb_err' % band)
            norm = self.__getattribute__('%s_offset' % band)
            irgb = rsp.math_utils.between(bins, norm,
                                          trgb + trgb_err * self.ags.factor[i])
            iagb = rsp.math_utils.between(bins,
                                          trgb - trgb_err * self.ags.factor[i],
                                          10.)

            nrgb = np.sum(hist[irgb])
            nagb = np.sum(hist[iagb])
            ratio_data['%s_ar_ratio' % band] = nagb / nrgb
            ratio_data['%s_ar_ratio_err' % band] = \
                rsp.utils.count_uncert_ratio(nagb, nrgb)
            ratio_data['n%s_rgb' % band] = nrgb
            ratio_data['n%s_agb'% band] = nagb
        return ratio_data



class DiagnosticPlots(Plotting):
    def __init__(self, vsfh):
        self.Plotting.__init__(vsfh)

    def plot_mass_met_table(self, opt_mass_met_file, ir_mass_met_file,
                            extra_str=''):
        from mpl_toolkits.axes_grid1 import ImageGrid
        from astroML.stats import binned_statistic_2d

        fig = plt.figure(figsize=(8, 8))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(2, 2),
                         axes_pad=.5,
                         add_all=True,
                         label_mode="all",
                         cbar_location="top",
                         cbar_mode="each",
                         cbar_size="7%",
                         cbar_pad="2%",
                         aspect=0)
        cmaps = [plt.cm.get_cmap('jet', 9), plt.cm.gray_r]
        #cmap =
        #cmap.set_bad('w', 1.)
        #fig, (axs) = plt.subplots(ncols=2, figsize=(8, 8), sharey=True)
        types = ['mean', 'count']
        k =-1
        for j in range(len(types)):
            for i, mass_met in enumerate([opt_mass_met_file,
                                          ir_mass_met_file]):
                k += 1
                with open(mass_met, 'r') as mmf:
                    lines = [l.strip() for l in mmf.readlines()
                             if not l.startswith('#')]

                mag = np.concatenate([np.array(l.split(), dtype=float)
                                      for l in lines[0::3]])
                mass = np.concatenate([np.array(l.split(), dtype=float)
                                       for l in lines[1::3]])
                mh = np.concatenate([np.array(l.split(), dtype=float)
                                     for l in lines[2::3]])

                N, xedges, yedges = binned_statistic_2d(mag, mass, mh,
                                                        types[j], bins=50)
                im = grid[k].imshow(N.T, origin='lower',
                               extent=[xedges[0], xedges[-1], yedges[0],
                                       yedges[-1]],
                               aspect='auto', interpolation='nearest',
                               cmap=cmaps[j])
                grid[k].cax.colorbar(im)
                #grid[i].cax.set_label('$[M/H]$')

        grid.axes_all[0].set_ylabel('${\\rm Mass}\ (M_\odot)$', fontsize=20)
        grid.axes_all[2].set_ylabel('${\\rm Mass}\ (M_\odot)$', fontsize=20)
        grid.axes_all[2].set_xlabel('$F814W$', fontsize=20)
        grid.axes_all[3].set_xlabel('$F160W$', fontsize=20)
        target = '_'.join(os.path.split(opt_mass_met_file)[1].split('_')[0:4])
        fig.suptitle('$%s$' % target.replace('_', '\ '), fontsize=20)
        plt.savefig('%s_mass_met%s.png' % (target, extra_str), dpi=150)
        return grid


