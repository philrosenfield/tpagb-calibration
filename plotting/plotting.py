import argparse
import difflib
import logging
import os
import sys

import matplotlib as mpl
import matplotlib.pylab as plt

import numpy as np
import ResolvedStellarPops as rsp
angst_data = rsp.angst_tables.angst_data

from ..TPAGBparams import EXT, snap_src, matchfake_loc, data_loc
from ..analysis.analyze import get_itpagb, parse_regions, get_trgb
from ..fileio import load_obs, find_fakes, find_match_param, load_lf_file
from ..fileio import load_observation
from ..utils import minmax
from ..pop_synth import stellar_pops
from ..sfhs import star_formation_histories

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

tpagb_model_default_color = '#156692'
data_tpagb_default_color = '#E1999E'
data_default_color = '#853E43'
model_default_color = '#01141F'

try:
    plt.style.use('presentation')
except:
    pass


def emboss(fg='w', lw=3):
    from matplotlib.patheffects import withStroke
    myeffect = withStroke(foreground=fg, linewidth=lw, alpha=0.5)
    return dict(path_effects=[myeffect])


def outside_labels(axs, fig=None, xlabel=None, ylabel=None, rows=True,
                   text_kw={}):
    """
    make an ndarray of axes only have lables on the outside of the grid
    also add common x and y label if fig, xlabel, ylabel passed.
    Returns
        np.ravel(axs)
    """
    ndim = len(np.shape(axs))
    #print(ndim)
    text_kw = dict({'ha': 'center', 'va': 'center', 'fontsize': 24}.items() + text_kw.items())
    if ndim == 1:
        if rows:
            bottom = axs[-1]
            [ax.tick_params(labelbottom=False, direction='in', which='both')
             for ax in np.ravel(axs)]
            bottom.tick_params(labelbottom=True)
        else:
            left = axs[0]
            [ax.tick_params(labelleft=False, direction='in', which='both')
             for ax in np.ravel(axs)]
            left.tick_params(labelleft=True)
    else:
        top = axs[0, :]
        bottom = axs[-1, :]
        left = axs[:, 0]
        right = axs[:, -1]

        [ax.tick_params(labelbottom=False, labelleft=False, direction='in',
                        which='both') for ax in np.ravel(axs)]
        [ax.tick_params(labeltop=True) for ax in top]
        [ax.tick_params(labelleft=True) for ax in left]
        [ax.tick_params(labelright=True) for ax in right]
        [ax.tick_params(labelbottom=True) for ax in bottom]

    axs = np.ravel(axs)

    if xlabel is not None:
        fig.text(0.5, 0.04, xlabel, **text_kw)

    if ylabel is not None:
        fig.text(0.06, 0.5, ylabel, rotation='vertical', **text_kw)
    return axs


def add_narratio_to_plot(ax, target, ratio_data, mid_txt='RGB'):
    stext_kw = dict({'color': model_default_color, 'fontsize': 14, 'ha': 'center'}.items() +
                    emboss().items())
    dtext_kw = dict(stext_kw.items() + {'color': data_default_color}.items())

    assert ratio_data[0]['target'] == 'data', \
        'the first line of the narratio file needs to be a data measurement'
    nagb = float(ratio_data[0]['nagb'])
    nrgb = float(ratio_data[0]['nrgb'])

    dratio = nagb / nrgb
    dratio_err = rsp.utils.count_uncert_ratio(nagb, nrgb)

    indx, = np.nonzero(ratio_data['target'] == target)
    mrgb = np.mean(np.array(ratio_data[indx]['nrgb'], dtype=float))
    magb = np.mean(np.array(ratio_data[indx]['nagb'], dtype=float))

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
    sagb_text = r'$\rm{R14}\ \langle N_{\rm TP-AGB}\rangle=%i$' % np.mean(magb)

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
    plt_kw = dict({'color': model_default_color}.items() + plt_kw.items())

    if agb_mod is not None:
        label = '${}$'.format(agb_mod)
        plt_kw_lab = dict(plt_kw.items() + {'label': label}.items())
    else:
        plt_kw_lab = plt_kw

    if ax is None:
        fig, (ax) = plt.subplots(figsize=(12, 6))

    if inorm is not None:
        ms = [mag2s[i][inorm[i]] for i in range(len(mag2s))]
    else:
        ms =[mag2s[i] * norms[i] for i in range(len(mag2s))]

    hists = [np.histogram(m, bins=bins)[0] for m in ms]
    minhists = np.min(np.array(hists).T, axis=1)
    maxhists = np.max(np.array(hists).T, axis=1)
    #meanhists = np.mean(np.array(hists).T, axis=1)
    meanhists = np.median(np.array(hists).T, axis=1)
    edges = np.repeat(bins, 2)
    fminhist = np.hstack((0, np.repeat(minhists, 2), 0))
    fmaxhist = np.hstack((0, np.repeat(maxhists, 2), 0))

    ax.fill_between(edges, fminhist, fmaxhist, color=plt_kw_lab['color'],
                    alpha='0.2')

    #ax.fill_between(bins[1:], minhists, maxhists, color=plt_kw_lab['color'],
    #                alpha='0.2')
    ax.plot(bins[1:], minhists, linestyle='steps', color=plt_kw_lab['color'],
            lw=2)
    ax.plot(bins[1:], maxhists, linestyle='steps', color=plt_kw_lab['color'],
            lw=2)
    l, = ax.plot(bins[1:], meanhists, linestyle='steps', lw=3, **plt_kw_lab)

    return ax, l


def plot_gal(mag2, bins, ax=None, target=None, plot_kw={}, fake_file=None,
             over_plot=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    target = target.replace('-', '\!-\!').upper()
    plot_kw = dict({'drawstyle': 'steps', 'lw': 1,
                    'color': data_default_color,
                    'label': '${}$'.format(target)}.items() + plot_kw.items())

    hist = np.histogram(mag2, bins=bins)[0]
    err = np.sqrt(hist)

    if fake_file is not None:
        comp_corr = stellar_pops.completeness_corrections(fake_file, bins)
        hist = np.histogram(mag2, bins=bins)[0]
        hist /= comp_corr[1:]
        err = np.sqrt(hist)
        plot_kw['lw'] += 1

    gl, = ax.plot(bins[1:], hist, **plot_kw)
    # mid bin
    ax.errorbar(bins[1:] - 0.05, hist, yerr=err, fmt='none',
                ecolor=plot_kw['color'])
    glt = None
    if over_plot is not None:
        hist = np.histogram(mag2[over_plot], bins=bins)[0]
        if fake_file is not None:
            hist /= comp_corr[1:]
        err = np.sqrt(hist)
        plot_kw['color'] = data_tpagb_default_color
        #plot_kw['label'] = '${}\ TP\!-\!AGB$'.format(target)
        plot_kw['label'] = '$TP\!-\!AGB$'.format(target)
        glt, = ax.plot(bins[1:], hist, **plot_kw)
        # mid bin
        ax.errorbar(bins[1:] - 0.05, hist, yerr=err, fmt='none',
                    ecolor=plot_kw['color'])

    return ax, gl, glt


def plot_models(lfd, bins, filt, maglimit=None, ax=None, plt_kw=None,
                agb_mod=None):
    plt_kw = plt_kw or {}
    mags = lfd[filt]

    ax, ml = plot_model(mag2s=mags, bins=bins, inorm=lfd['idx_norm'],
                        maglimit=maglimit, ax=ax, plt_kw=plt_kw, agb_mod=agb_mod)

    plt_kw['color'] = tpagb_model_default_color
    ax, mlt = plot_model(mag2s=mags, bins=bins, inorm=lfd['sim_agb'],
                        maglimit=maglimit, ax=ax, plt_kw=plt_kw,
                        agb_mod='TP\!-\!AGB')
                        #agb_mod='{}\ TP\!-\!AGB'.format(agb_mod))

    return ax, ml, mlt


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
                col1='MAG2_ACS', col2='MAG4_IR', dmag=0.1, extra_str='',
                match_param=None):
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
        mag1, mag2 = load_observation(observation, col1, col2,
                                      match_param=match_param)
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
                            linestyle='steps')
            axs[1].plot(Mbins[1:], dmdiff, drawstyle='steps')
            axs[2].plot(Mbins[1:], sdiff, drawstyle='steps')

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
    fig1.savefig('{}_{}_comp_lfs{}'.format(extra_str, filter1, EXT))
    fig2.savefig('{}_{}_comp_lfs{}'.format(extra_str, filter2, EXT))
    return opt_axs, nir_axs


def compare_to_gal(lf_file, observation, filter1='F814W_cor',
                   filter2='F160W_cor', col1='MAG2_ACS', col2='MAG4_IR',
                   dmag=0.1, narratio_file=None, make_plot=True,
                   regions_kw=None, xlims=[None, None], ylims=[None, None], extra_str='',
                   agb_mod=None, mplt_kw={}, dplot_kw={},
                   match_param=None):
    '''
    Plot the LFs and galaxy LF.

    ARGS:
    narratio: overlay NRGB, NAGB, and NAGB/NRGB +/- err
    no_agb: plot the LF without AGB stars

    RETURNS:
    ax1, ax2: axes instances created for the plot.

    '''
    agb_mod = translate_agbmod(agb_mod)
    target = os.path.split(lf_file)[1].split('_')[0]
    ir_mtrgb = get_trgb(target, filter2='F160W', filter1=None)
    opt_mtrgb = get_trgb(target, filter2='F814W')
    trgb_color = opt_mtrgb - ir_mtrgb

    mag1, mag2 = load_observation(observation, col1, col2,
                                  match_param=match_param)
    data_tpagb = get_itpagb(target, mag1 - mag2, mag2, col2,
                            off=trgb_color)
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

    lfd = load_lf_file(lf_file)
    for filt, fake_file in zip([filter1, filter2], [optfake, nirfake]):
        if filt == filter1:
            mag = mag1
            xlim = xlims[0]
            xlim[-1] = angst_data.get_50compmag(target.replace('-', '_').upper(),
                                                'F814W')
            ylim = ylims[0]
        else:
            mag = mag2
            xlim = xlims[1]
            xlim[-1] = angst_data.get_snap_50compmag(target.upper(), 'F160W')
            ylim = ylims[1]

        bins = np.arange(mag.min(), mag.max(), step=dmag)

        ax, ml, mlt = plot_models(lfd, bins, filt, plt_kw=mplt_kw, agb_mod=agb_mod)

        # plot galaxy data
        ax, gl, glt = plot_gal(mag, bins, ax=ax, target=target, fake_file=fake_file,
                      plot_kw=dplot_kw, over_plot=data_tpagb)

        ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(1, ax.get_ylim()[-1])
        if xlim is not None:
            ax.set_xlim(xlim)

        band = 'nir'
        if '814' in filt:
            band = 'opt'
        ax = add_trgb(ax, target, band, regions_kw=regions_kw)

        #ax.legend(loc='center left', handles=[ml, gl, mlt, glt], ncol=2)
        lab = r'$\rm{{{}}}$'.format(target.upper().replace('-','\!-\!'))
        ax.text(0.98, 0.05, lab, ha='right', fontsize=20, transform=ax.transAxes,
                **emboss())
        ax.set_xlabel('${}$'.format(filt.replace('_cor', '')), fontsize=20)
        ax.set_ylabel('${\#}$', fontsize=20)
        if narratio_file is not None:
            ax = add_narratio_to_plot(ax, target, ratio_data, mid_txt='RGB')

        plt.tick_params(labelsize=16)
        outfile = '{}_{}{}_lfs{}'.format(lf_file.split('_lf')[0], filt, extra_str, EXT)
        plt.savefig(outfile)
        logger.info('wrote {}'.format(outfile))
    return


def add_trgb(ax, target, band, lf=True, regions_kw=None):
    offset = 1.
    #opt_fake, nir_fake = find_fakes(target)
    #comp1, comp2 = stellar_pops.limiting_mag(nir_fake, 0.9)
    faint_color = 'k'

    if not 'ir' in band.lower():
        trgb = angst_data.get_tab5_trgb_av_dmod(target.upper())[0]
        trgb_exclude = 0.1
        magbright = trgb + trgb_exclude
        magfaint = trgb + offset
        regions_kw = {'offset': offset,
                      'trgb_exclude': trgb_exclude,
                      'trgb': trgb,
                      'mag_bright': magbright,
                      'mag_faint': magfaint}


    ax = add_lines_to_plot(ax, faint_color=faint_color, lf=lf,
                           **regions_kw)
    return ax

def add_lines_to_plot(ax, mag_bright=None, mag_faint=None, offset=0.,
                      trgb=None, trgb_exclude=0., lf=True, col_min=None,
                      col_max=None, faint_color=None, **kwargs):

    #if mag_bright is not None:
    #    mid = mag_bright
    #else:
    #    mid = trgb + trgb_exclude
    faint_color = faint_color or 'black'

    if mag_faint is not None:
        low = mag_faint
    else:
        low = trgb + offset

    yarr = np.linspace(*ax.get_ylim())
    if lf:
        # vertical lines around the trgb exclude region
        if trgb_exclude > 0:
            ax.fill_betweenx(yarr, trgb - trgb_exclude, trgb + trgb_exclude,
                             color='black', alpha=0.1)
        ax.vlines(trgb, *ax.get_ylim(), color='black', linestyle='--')
        if offset > 0 or mag_faint is not None:
            ax.vlines(low, *ax.get_ylim(), color=faint_color, linestyle='--')
        #ax.fill_betweenx(yarr, mid, low, color='black', alpha=0.1)
        #if mag_limit_val is not None:
        #    ax.fill_betweenx(yarr, mag_limit_val, ax.get_xlim()[-1],
        #                     color='black', alpha=0.5)
    else:
        xarr = np.linspace(*ax.get_xlim())

        # vertical lines around the trgb exclude region
        if trgb_exclude > 0:
            ax.fill_between(yarr, trgb - trgb_exclude, trgb + trgb_exclude,
                             color='black', alpha=0.1)
        ax.axhline(trgb, color='black', linestyle='--')
        if offset > 0 or mag_faint is not None:
            ax.axhline(low, color='black', linestyle='--')
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

        outfile = '{}_{}_{}{}'.format(outfmt, zstr, band, EXT)
        plt.savefig(outfile)
        logger.info('wrote {}'.format(outfile))
    return

def translate_agbmod(agb_mod):
    if 'm36' in agb_mod:
        agb_mod = 'R14'
    return agb_mod

def main(argv):
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

    parser.add_argument('-t', '--trgb', type=str, default=None,
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
    main(sys.argv[1:])
