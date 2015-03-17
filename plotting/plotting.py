import argparse
import sys
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import ResolvedStellarPops as rsp
import sys
from ..sfhs import star_formation_histories
from ..pop_synth import stellar_pops
from ..fileio import load_obs, find_fakes


# where the matchfake files live
from ..TPAGBparams import snap_src


optfilter2 = 'F814W'
# optfilter1 varies from F475W, F555W, F606W
nirfilter2 = 'F160W'
nirfilter1 = 'F110W'

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


def load_lf_file(lf_file):
    header = open(lf_file).readline().replace('#', '').split()
    try:
        header[header.index('F814W')] = 'optfilter2'
        header[header.index('F110W')] = 'nirfilter1'
        header[header.index('F160W')] = 'nirfilter2'
    except:
        header[header.index('F814W_cor')] = 'optfilter2'
        header[header.index('F110W_cor')] = 'nirfilter1'
        header[header.index('F160W_cor')] = 'nirfilter2'

    filt1, = [h for h in header if h.upper().startswith('F')]
    header[header.index(filt1)] = 'optfilter1'
    
    ncols = len(header)
    with open(lf_file, 'r') as lf:
        lines = [l.strip() for l in lf.readlines() if not l.startswith('#')]

    lfd = {}
    for i, key in enumerate(header):
        lfd[key] = [np.array(map(rsp.utils.is_numeric, l.split()))
                    for l in lines[i::ncols]]
    opt_dict = {k: v for k, v in lfd.items() if k.startswith('opt')}
    nir_dict = {k: v for k, v in lfd.items() if k.startswith('nir')}

    return opt_dict, nir_dict


def plot_random_sfhs(vsfh):
    '''
    plot the random sfr arrays with the data sfr arrays.
    Hard coded file locations. So read the code.
    '''
    sfr_file_loc = vsfh.outfile_loc
    hmc_file_loc = vsfh.base

    outfile = os.path.join(sfr_file_loc, '%s_random_sfr.png' % vsfh.target)
    hmc_file, = rsp.fileIO.get_files(hmc_file_loc, '%s*mcmc.zc' % vsfh.target)

    sfh = star_formation_histories(hmc_file, 'match-hmc',
                                   sfr_file_loc=sfr_file_loc,
                                   sfr_file_search_fmt='*sfr')

    sfh.plot_sfh('sfr', plot_random_arrays_kw={'from_files': True},
                 outfile=outfile)
    return


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

    def plot_by_stage(self, ax1, ax2, add_stage_lfs='default', stage_lf_kw=None,
                      cols=None, trilegal_output=None, hist_it_up=False,
                      narratio=True):

        if add_stage_lfs == 'all':
            add_stage_lfs = ['MS', 'RGB', 'HEB', 'RHEB',
                             'BHEB', 'EAGB', 'TPAGB']
        if add_stage_lfs == 'default':
            add_stage_lfs = ['RGB', 'EAGB', 'TPAGB']

        nstages = len(add_stage_lfs)
        stage_lf_kw = stage_lf_kw or {}
        stage_lf_kw = dict({'linestyle': 'steps', 'lw': 2}.items() +
                           stage_lf_kw.items())
        if hasattr(stage_lf_kw, 'label'):
            stage_lf_kw['olabel'] = stage_lf_kw['label']

        # load the trilegal catalog if it is given, if it is given,
        # no LF scaling... need to save this info better. Currently only
        # in log files.
        if trilegal_output is not None:
            self.files.read_trilegal_catalog(trilegal_output,
                                             filter1=get_filter1(self.target))
            self.files.load_trilegal_data()
            self.opt_norm = 1.
            self.ir_norm = 1.

        self.files.load_data_for_normalization(target=self.target, ags=self.ags)
        assert hasattr(self.files, 'opt_mag'), \
            'Need opt_mag or trilegal_output'

        for ax, mag, norm, sinds, bins in \
            zip([ax1, ax2],
                [self.files.sgal.data.get_col('F814W')-self.files.opt_moffset,
                 self.files.sgal.data.get_col('F160W')-self.files.ir_moffset],
                [self.opt_norm, self.ir_norm],
                [self.files.opt_color_cut, self.files.ir_color_cut],
                [self.files.opt_bins, self.files.ir_bins]):

            #self.files.sgal.make_lf(mag, stages=add_stage_lfs, bins=bins,
            #                        inds=sinds, hist_it_up=hist_it_up)
            self.files.sgal.make_lf(mag, stages=add_stage_lfs, bins=bins,
                                    hist_it_up=hist_it_up)
            #import pdb
            #pdb.set_trace()
            k = 0
            for i in range(nstages):
                istage = add_stage_lfs[i].lower()
                try:
                    hist = self.files.sgal.__getattribute__('i%s_lfhist' %
                                                            istage)
                except AttributeError:
                    continue
                # combine all HeB stages into one for a cleaner plot.
                if istage == 'heb':
                    hist = \
                    np.sum([self.files.sgal.__getattribute__('i%s_lfhist' %
                                                             jstage.lower())
                            for jstage in add_stage_lfs
                            if 'heb' in jstage.lower()], axis=0)
                elif 'heb' in istage:
                    continue

                bins = \
                self.files.sgal.__getattribute__('i%s_lfbins' %
                                                 istage)
                stage_lf_kw['color'] = cols[k]
                k += 1
                stage_lf_kw['label'] = '$%s$' % istage.upper().replace('PA', 'P\!-\!A')
                norm_hist = hist # * norm
                norm_hist[norm_hist < 3] = 3
                ax.plot(bins[:-1], norm_hist, **stage_lf_kw)

        sopt_hist, sopt_bins = self.files.sgal.make_lf(self.files.opt_mag,
                                                      bins=self.files.opt_bins,
                                                      hist_it_up=hist_it_up)
        sir_hist, sir_bins = self.files.sgal.make_lf(self.files.ir_mag,
                                                     bins=self.files.ir_bins,
                                                     hist_it_up=hist_it_up)
        # 3 is the plot limit...
        sopt_hist[sopt_hist < 3] = 3
        sir_hist[sir_hist < 3] = 3
        stage_lf_kw['color'] = 'black'

        if narratio is False:
            if not hasattr(stage_lf_kw, 'olabel'):
                lab = '$Total$'
                #if hasattr(self, 'agb_mod'):
                #    model = self.agb_mod.split('_')[-1]
                #    lab = model_plots.translate_model_name(model, small=True)
                stage_lf_kw['label'] = lab
            else:
                stage_lf_kw['label'] = stage_lf_kw['olabel']
        ax1.plot(sopt_bins[:-1], sopt_hist, **stage_lf_kw)
        ax2.plot(sir_bins[:-1], sir_hist, **stage_lf_kw)
        return ax1, ax2

    def compare_to_gal(self):
        compare_to_gal(**self.__dict__)


def add_narratio_to_plot(ax, target, ratio_data, mid_txt='RGB', opt=True):
    stext_kw = dict({'color': 'black', 'fontsize': 14, 'ha': 'center'}.items() +
                    rsp.graphics.GraphicsUtils.ann_kwargs.items())
    dtext_kw = dict(stext_kw.items() + {'color': 'darkred'}.items())

    if opt:
        filt = 'opt'
    else:
        filt = 'nir'

    assert ratio_data[0]['target'] == 'data', \
        'the first line of the narratio file needs to be a data measurement'
    nagb = ratio_data[0]['{}nagb'.format(filt)]
    nrgb = ratio_data[0]['{}nrgb'.format(filt)]

    dratio = nagb / nrgb
    dratio_err = rsp.utils.count_uncert_ratio(float(nagb), float(nrgb))

    indx, = np.nonzero(ratio_data['target'] == target)
    mrgb = ratio_data[indx]['{}nrgb'.format(filt)]
    magb = ratio_data[indx]['{}nagb'.format(filt)]

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
    sratio_text = '$f=%.3f\pm%.3f$' % (np.mean(ratio_data['{}ar_ratio'.format(filt)]),
                                       np.mean(ratio_data['{}ar_ratio_err'.format(filt)]))

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
    '''needs work, but: plot the lf files.'''
    # set up the plot
    plt_kw = dict({'linestyle': 'steps-mid', 'color': 'black',
                   'alpha': 0.2}.items() + plt_kw.items())
    if agb_mod is not None:
        label = r'$%s$' % agb_mod.split('_')[-1]
        plt_kw_lab = dict(plt_kw.items() + {'label': label}.items())
    else:
        plt_kw_lab = plt_kw
    if ax is None:
        fig, (ax) = plt.subplots(figsize=(12, 6))
        #plt.subplots_adjust(right=0.95, left=0.05)

    for i in range(len(mag2s)):
        if inorm is not None:
            mag2 = mag2s[i][inorm[i]]
            norm = 1.
        else:
            mag2 = mag2s[i][inds]
            norm = norms[i]

        if maglimit is not None:
            inds, = np.nonzero(mag2 <= maglimit)
        else:
            inds = np.arange(len(mag2))

        hist = np.histogram(mag2[inds], bins=bins)[0]
        if i != 0:
            kw = plt_kw
        else:
            kw = plt_kw_lab
            pass
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


def plot_models(lf_file, bins, opt=True, maglimit=None, ax=None, plt_kw=None,
                agb_mod=None):
    plt_kw = plt_kw or {}
    opt_lfd, nir_lfd = load_lf_file(lf_file)
    if opt:
        mag2s = opt_lfd['optfilter2']
        inorm = opt_lfd['optidx_norm']
    else:
        mag2s = nir_lfd['nirfilter2']
        inorm = nir_lfd['niridx_norm']

    ax = plot_model(mag2s=mag2s, bins=bins, inorm=inorm,
                    maglimit=maglimit, ax=ax, plt_kw=plt_kw, agb_mod=agb_mod)
    return ax


def compare_to_gal(optfake=None, nirfake=None, optfilter1=None, target=None,
                   lf_file=None, optfilter2_limit=None, nirfilter2_limit=None,
                   draw_lines=True, xlim=None, ylim=None, extra_str='',
                   narratio_file=None, ast_cor=False, agb_mod=None,
                   mplt_kw={}, dplot_kw={}, optregions_kw={}, nirregions_kw={}):
    '''
    Plot the LFs and galaxy LF.

    ARGS:
    narratio: overlay NRGB, NAGB, and NAGB/NRGB +/- err
    no_agb: plot the LF without AGB stars

    RETURNS:
    ax1, ax2: axes instances created for the plot.

    '''
    if narratio_file is not None:
        ratio_data = rsp.fileio.readfile(narratio_file, string_column=[0, 1, 6])

    optgal, nirgal = load_obs(target, optfilter1=optfilter1)

    for opt in [True, False]:
        if opt:
            maglimit = optfilter2_limit
            fake_file = optfake
            extra_str += '_opt'
            regions_kw = optregions_kw
            try:   
                mag2 = optgal.data['MAG2_ACS']
            except:
                mag2 = optgal.data['MAG2_WFPC2']
            filter2 = optfilter2
        else:
            maglimit = nirfilter2_limit
            fake_file = nirfake
            mag2 = nirgal.data['MAG2_IR']
            extra_str = extra_str.replace('opt', 'nir')
            regions_kw = nirregions_kw
            filter2 = nirfilter2

        if ast_cor:
            if not '_ast_cor' in extra_str:
                extra_str += '_ast_cor'

        bins = np.arange(16, 27, 0.1)
        ax = plot_models(lf_file, bins, opt=opt, maglimit=maglimit, plt_kw=mplt_kw)

        # plot galaxy data
        ax = plot_gal(mag2, bins, ax=ax, target=target, fake_file=fake_file,
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

        if draw_lines:
            ax = add_lines_to_plot(ax, **regions_kw)
        
        ax.legend(loc='lower right')
        ax.set_xlabel('${}$'.format(filter2), fontsize=20)

        if narratio_file is not None:
            ax = add_narratio_to_plot(ax, target, ratio_data, mid_txt='RGB',
                                      opt=opt)

        plt.tick_params(labelsize=16)
        outfile = '{}{}_lfs.png'.format(lf_file.split('_lf')[0], extra_str)
        plt.savefig(outfile)
        print 'wrote {}'.format(outfile)
    return

def add_lines_to_plot(ax, mag_bright=None, mag_faint=None, offset=None, trgb=None,
                      trgb_exclude=None, **kwargs):
    if mag_bright is not None:
        low = mag_faint
        mid = mag_bright
    else:
        assert offset is not None, \
        'need either offset or mag limits'
        low = trgb + offset
        mid = trgb + trgb_exclude
    
    yarr = np.linspace(*ax.get_ylim())
    
    # vertical lines around the trgb exclude region
    ax.fill_betweenx(yarr, trgb - trgb_exclude, trgb + trgb_exclude,
                     color='black', alpha=0.1)
    ax.vlines(trgb, *ax.get_ylim(), color='black', linestyle='--')
    
    ax.vlines(low, *ax.get_ylim(), color='black', linestyle='--')
    #ax.fill_betweenx(yarr, mid, low, color='black', alpha=0.1)
    #if mag_limit_val is not None:
    #    ax.fill_betweenx(yarr, mag_limit_val, ax.get_xlim()[-1],
    #                     color='black', alpha=0.5)
    return ax


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
    
    parser.add_argument('-n', '--narratio_file', type=str, help='model narratio file')

    parser.add_argument('target', type=str, help='target name')

    parser.add_argument('lf_file', type=str, help='model LFs file')

    args = parser.parse_args(argv)

    optregions_kw, nirregions_kw = parse_regions(args)
    ast_cor = 'ast' in args.lf_file

    optfake, nirfake = find_fakes(args.target)

    compare_to_gal(optfake=optfake, nirfake=nirfake, optfilter1=args.optfilter1,
                   target=args.target, lf_file=args.lf_file, 
                   narratio_file=args.narratio_file, ast_cor=ast_cor, agb_mod=None,
                   optregions_kw=optregions_kw, nirregions_kw=nirregions_kw,
                   mplt_kw={}, dplot_kw={},
                   optfilter2_limit=None,
                   nirfilter2_limit=None,
                   draw_lines=True, xlim=None, ylim=None)

if __name__ == '__main__':
    main(sys.argv[1:])