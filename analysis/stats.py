import argparse
import logging
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import ResolvedStellarPops as rsp

#from TPAGBparams import snap_src

from astroML.stats import binned_statistic_2d
from ..fileio import load_photometry
from .analyze import get_itpagb
from ..utils import minmax
from ..plotting.plotting import compare_to_gal
logger = logging.getLogger()


def chi2(obs, model):
    from scipy import stats
    # t-test:
    #ti = (ohist - mhist) ** 2 / (ohist + mhist)
    #naners = np.isnan(ti)
    #ti[naners] = 0
    #print np.sum(ti)
    # maybe there is a better way to mask this... chiw was a typo...
    import pdb; pdb.set_trace()
    chiw = (model - obs) ** 2 / obs

    naners = np.isinf(chiw)
    chiw[np.isinf(chiw)] = 0
    naners = np.isnan(chiw)
    chiw[np.isnan(chiw)] = 0

    N = len(np.nonzero(chiw > 0)[0]) - 1

    chisq = np.sum(chiw) / float(N)
    pval = 1. - stats.chi2.cdf(chisq, N)
    return chisq, pval

def hess_stats(target, color, ymag, scolor, symag, dcol, dmag, nmodels=1):
    # apply cmd cuts
    # just the tpagb:
    itpagb = get_itpagb(target, color, ymag)
    sitpagb = get_itpagb(target, scolor, symag)

    # the entire cmd:
    iall = np.arange(len(color))
    siall = np.arange(len(scolor))

    # the cmd without tpagb:
    icmd = [a for a in iall if not a in itpagb]
    sicmd = [a for a in siall if not a in sitpagb]
    # could also single out non-rheb and non-tpagb ...

    probs = []
    extents = []
    Zs = []

    for i, (inds, sinds) in enumerate(zip([itpagb, iall, icmd],
                                          [sitpagb, siall, sicmd])):
        if len(inds) > 0:
            cbins = np.arange(*minmax(scolor[sinds], color[inds]), step=dcol)
            mbins = np.arange(*minmax(symag[sinds], ymag[inds]), step=dmag)

            shess, xe, ye = binned_statistic_2d(scolor[sinds], symag[sinds],
                                                symag[sinds], 'count',
                                                bins=[cbins, mbins])

            # get the mean hess if many models are co-added.
            shess /= float(nmodels)

            hess, xe, ye = binned_statistic_2d(color[inds], ymag[inds], ymag[inds],
                                               'count', bins=[cbins, mbins])
            extent = [xe[0], xe[-1], ye[-1], ye[0]]
            _, pct_dif, sig = rsp.match.likelihood.stellar_prob(hess, shess)
            #import pdb; pdb.set_trace()
            prob = np.sum(np.abs(pct_dif[np.isfinite(pct_dif)]))/float(len(hess.flatten()))
            dif = hess - shess
        else:
            prob = np.nan
            extent = [0, 1, 0, 1]
            hess, shess, pct_dif, sig = np.array([]), np.array([]), np.array([]), np.array([])
        print prob
        probs.append(prob)
        extents.append(extent)
        Zs.append([hess.T, shess.T, pct_dif.T, sig.T])

    return probs, extents, Zs

def compare_hess(lf_file, observation, filter1='F814W_cor', filter2='F160W_cor',
                col1='MAG2_ACS', col2='MAG4_IR', dcol=0.1, dmag=0.1,
                yfilter='I', narratio_file=None, make_plot=True,
                outfile='compare_hess.txt', flatten=True, agb_mod='agb_mod'):

    if not os.path.isfile(outfile):
        header = '# target {0}_prob {0}_tpagb_prob {0}_notpagb_prob nmodels \n'.format(agb_mod.split('_')[-1])
        with open(outfile, 'w') as out:
            out.write(header)
    linefmt = '{:15s} {:.3f} {:.3f} {:.3f} {} \n'

    target = os.path.split(lf_file)[1].split('_')[0]

    # load data
    color, ymag, scolor, symag, nmodels = \
        load_photometry(lf_file, observation, filter1=filter1, filter2=filter2,
                         col1=col1, col2=col2, yfilter=yfilter,
                         comp_frac=0.9, flatten=flatten)
    if flatten:
        color = [color]
        ymag = [ymag]
        scolor = [scolor]
        symag = [symag]
        nmodels = 1

    for i in range(len(color)):
        probs, extents, Zs = hess_stats(target, color[i], ymag[i], scolor[i], symag[i], dcol,
                                        dmag, nmodels=nmodels)

        line = linefmt.format(target, probs[1], probs[0], probs[2], nmodels)
        with open(outfile, 'a') as out:
            out.write(line)
        logger.info('wrote to {}'.format(outfile))

    if make_plot:
        fmt0 = r'$N_{{\rm TP-AGB}}={:.0f}$'
        fmt1 = r'$N_{{\rm RGB}}={:.0f}\ \frac{{N_{{TP-AGB}}}}{{N_{{RGB}}}}={:.3f}\pm{:.3f}$'
        labels = ['data', 'model', r'$\% diff.$', 'sig.']

        if narratio_file is not None:
            nrgb, nagb, dratio, dratio_err, mrgb, magb, mratio, mratio_err = \
                narratio(narratio_file, target)
            labels[0] = fmt1.format(nrgb, dratio, dratio_err)
            labels[1] = fmt1.format(mrgb, mratio, mratio_err)


        figname = lf_file.replace('.dat', '.png')
        fignames = [figname.replace('lf', 'tpagb_hess'),
                    figname.replace('lf', 'hess'),
                    figname.replace('lf', 'notpagb_hess')]

        for i in range(len(probs)):
            labels[-2] = r'$mean \% diff={:.2f}$'.format(probs[i])
            grid = rsp.match.graphics.match_plot(Zs[i], extents[i], labels=labels)
            [ax.set_ylabel(r'$\rm F160W$') for ax in grid.axes_column[0]]
            [ax.set_xlabel(r'$\rm F160W - F814W$') for ax in grid.axes_row[1]]
            plt.savefig(fignames[i])


# 1d analysis - per galaxy
# see compare_to_gal in plotting.py
# bin into LF
# if lf file has many cmds, do mean, median or whatever
# compare to observation

# 2d analysis - all galaxies
# chi2 vs galaxy plot for each agb_mod

# 1d analysis - all galaxies
# see compare_lfs in plotting.py

# chi2 vs galaxy plot for each agb_mod

def narratio(narratio_file, target, nagb=None, magb=None):
    ratio_data = rsp.fileio.readfile(narratio_file,
                                     string_column=[0, 1, 2])
    if nagb is None:
        nagb = float(ratio_data[0]['nagb'])
    nrgb = float(ratio_data[0]['nrgb'])

    dratio = nagb / nrgb
    dratio_err = rsp.utils.count_uncert_ratio(nagb, nrgb)

    indx, = np.nonzero(ratio_data['target'] == target)
    mrgb = np.mean(map(float, ratio_data[indx]['nrgb']))

    if magb is None:
        magb = np.mean(map(float, ratio_data[indx]['nagb']))

    mratio = magb / mrgb
    mratio_err = rsp.utils.count_uncert_ratio(magb, mrgb)



    return nrgb, nagb, dratio, dratio_err, mrgb, magb, mratio, mratio_err


def main2(argv):
    description = ("stats...")
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-c', '--colnames', type=str, default='MAG2_ACS,MAG4_IR',
                        help='comma separated column names in observation data')

    parser.add_argument('-s', '--scolnames', type=str, default='F814W_cor,F160W_cor',
                        help='comma separated column names in trilegal catalog')

    parser.add_argument('-d', '--dcolmag', type=str, default='0.05,0.1',
                        help='comma separated dcol, dmag for binning')

    parser.add_argument('-n', '--narratio_file', type=str,
                        help='nagb/nrgb ratio file')

    parser.add_argument('-y', '--yfilter', type=str, default='I',
                        help='V or I filter to use as y axis of CMD')

    parser.add_argument('-f', '--flatten', action='store_false',
                        help='treat model LFs individually (not mean)')

    parser.add_argument('-p', '--make_plot', action='store_false',
                        help='do not save plots')

    parser.add_argument('-o', '--outfile', type=str, default='default',
                        help='outfile name')

    parser.add_argument('lf_file', type=str,
                        help='luminosity function file')

    parser.add_argument('observation', type=str,
                        help='data file to compare to')

    parser.add_argument('agb_mod', type=str,
                        help='agb model name')


    args = parser.parse_args(argv)
    print args
    col1, col2 = args.colnames.split(',')
    filter1, filter2 = args.scolnames.split(',')
    dcol, dmag = map(float, args.dcolmag.split(','))
    if args.outfile == 'default':
        args.outfile = 'compare_hess_{}.txt'.format(args.agb_mod)
    else:
        args.outfile = args.outfile.strip()

    compare_hess(args.lf_file, args.observation, filter1=filter1,
                 filter2=filter2, col1=col1, col2=col2, dcol=dcol, dmag=dmag,
                 yfilter=args.yfilter, narratio_file=args.narratio_file,
                 outfile=args.outfile, agb_mod=args.agb_mod, flatten=args.flatten,
                 make_plot=args.make_plot)

    # just use plotting.main...
    #compare_to_gal(args.lf_file, args.observation, filter1=filter1,
    #               filter2=filter2, col1=col1, col2=col2,
    #               dmag=dmag, narratio_file=args.narratio_file, make_plot=True,
    #               regions_kw=None, agb_mod=args.agb_mod)

def main(argv):
    parser = argparse.ArgumentParser(description='make narratio table')

    parser.add_argument('-n', '--narratio_file', type=str,
                        help='nagb/nrgb ratio file')

    parser.add_argument('-s', '--search_str', type=str, default='*nar*dat')


    args = parser.parse_args(argv)

    if args.narratio_file:
        assert os.path.isfile(args.narratio_file), 'file not found'
        ratios = [args.narratio_file]
    else:
        ratios = rsp.fileio.get_files(os.getcwd(), args.search_str)
        if len(ratios) == 0:
            print('files not found {}'.format(os.path.join(os.getcwd(), args.search_str)))

    narratio_table(ratios)

def chi2plot(chi2table, outfile_loc=None, flatten=True):
    """
    this works by pasting tables together and deleting the target names
    (except for the first column) and nmodel columns.

    this is being used in ipython ...
    """
    if outfile_loc is None:
        outfile_loc = os.getcwd()

    chi2tab = rsp.fileio.readfile(chi2table, string_column=0)
    probs = [c for c in chi2tab.dtype.names if c.endswith('prob')]
    agb_mods = list(np.unique([c.split('_')[0] for c in probs]))
    nagb_mods = len(agb_mods)

    cols = [u'#E24A33', u'#348ABD', u'#988ED5', u'#777777', u'#FBC15E',
            u'#8EBA42', u'#FFB5B8']

    all_targets = chi2tab['target']
    targets, uinds = np.unique(chi2tab['target'], True)

    offsets = np.linspace(0, 1, len(targets))
    tnames = ['$%s$' % t.replace('-deep', '').replace('-halo-6', '').replace('-', '\!-\!').upper() for t in targets]
    nmeasurements = False
    if len(targets) * nagb_mods > len(all_targets):
        uinds = np.append(uinds, len(chi2tab['target']))
        isets = [np.arange(uinds[i], uinds[i+1]) for i in range(len(uinds)-1)]
        if flatten:
            print 'take a mean'
        else:
            nmeasurements = True
            assert len(cols) >= len(targets), 'need more colors!'
            code
    else:
        assert len(cols) >= nagb_mods, 'need more colors!'

    # key plots still missing:
    # chi2 versus fraction of y<1 Gyr ages? and agains metallicity?
    # or against both, e.g chi2 X age with dots coloured according to metallicity

    ycols = []#[y for y in chi2tab.dtype.names[1:] if not y.endswith('prob')]
    for ycol in ycols:
        fig, axs = plt.subplots(ncols=3, sharex=True, sharey=False,
                                figsize=(15, 6))
        isort = np.argsort(chi2tab[ycol])
        try:
            axs[0].plot(chi2tab['chi2eff'][isort], chi2tab[ycol][isort], 'o', color='k',
                        label='chi2eff')
        except:
            pass
        for colmn in probs:
            col = cols[agb_mods.index(colmn.split('_')[0])]
            iax = 0
            if 'tpagb' in colmn:
                iax = 1
            if 'not' in colmn:
                iax = 2
            ax = axs[iax]
            ax.plot(chi2tab[colmn][isort], chi2tab[ycol][isort], 'o', color=col,
                    label=colmn.split('_')[0])
        for i in range(len(targets)):
            ax.annotate(tnames[i], (0, chi2tab[ycol][i]))
        [ax.set_xlabel('$\% diff$', fontsize=20) for ax in axs]
        [ax.set_ylabel('${}$'.format(ycol), fontsize=20) for ax in axs]

        axs[0].legend(loc=0, numpoints=1)
        axs[1].annotate(r'$\rm{TP\!-\!AGB\ Only}$', (0.02, 0.02), fontsize=16,
                        xycoords='axes fraction')
        axs[2].annotate(r'$\rm{No TP\!-\!AGB}$', (0.02, 0.02), fontsize=16,
                        xycoords='axes fraction')
        outfile = os.path.join(outfile_loc, 'chi2_{}.png'.format(ycol))
        fig.savefig(outfile, dpi=150)

    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=False,
                            figsize=(15, 6))

    for ioff, colmn in enumerate(probs):
        col = cols[agb_mods.index(colmn.split('_')[0])]
        if nmeasurements:
            col = cols[target.index(3)] ###
        iax = 0
        if 'tpagb' in colmn:
            iax = 1
        if 'not' in colmn:
            iax = 2
        ax = axs[iax]

        #ax.errorbar(offsets[ioff], val, yerr=errval, marker=sym, color=col, ms=12,
        #            mfc=mfc, ecolor='black', mew=1.5, elinewidth=2)
        ax.scatter(offsets, chi2tab[colmn], marker='o', s=60, c=col, alpha=0.5,
                   edgecolors='none')

        ax.hlines(np.median(chi2tab[colmn]), -0.1, 1.1, color=col,
                  label=colmn.split('_')[0], alpha=0.5, lw=2)

        ax.xaxis.set_ticks(offsets)
        ax.set_xticklabels(tnames)
        [t.set_rotation(30) for t in ax.get_xticklabels()]
    #plt.tick_params(labelsize=16)
    fig.subplots_adjust(hspace=0.1, bottom=0.15, left=0.1, right=0.95)

    xlims = ax.get_xlim()
    off = np.diff(offsets)[0]
    ax.set_xlim(xlims[0] - off / 2, xlims[1] + off / 2)
    [ax.set_ylabel('$\% diff$', fontsize=20) for ax in axs]
    #[ax.set_ylim(0, 25) for ax in axs[:, 0]]
    #[ax.set_ylim(0, 10) for ax in axs[:, 1]]


    axs[0].legend(loc=0, numpoints=1)
    axs[1].annotate(r'$\rm{TP\!-\!AGB\ Only}$', (0.02, 0.02), fontsize=16,
                    xycoords='axes fraction')

    outfile = os.path.join(outfile_loc, 'chi2_target.png')
    fig.savefig(outfile, dpi=150)

    return axs


def narratio_table(nartables):
    line = ''
    for i, nartable in enumerate(nartables):
        ratio_data = rsp.fileio.readfile(nartable, string_length=36,
                                         string_column=[0, 1, 2])
        targets = np.unique([t for t in ratio_data['target'] if not 'data' in t])
        fmt = r'${:.3f}\pm{:.3f}$'
        line += '% '+ nartable
        line += '\n'
        print 'target data_ratio model_ratio frac_diff'
        #import pdb; pdb.set_trace()
        for target in targets:
            dindx = np.nonzero(ratio_data['target'] == 'data'.format(target))[0][0]
            indx, = np.nonzero(ratio_data['target'] == target)

            nagb = float(ratio_data[dindx]['nagb'])
            nrgb = float(ratio_data[dindx]['nrgb'])

            dratio = nagb / nrgb
            dratio_err = rsp.utils.count_uncert_ratio(nagb, nrgb)

            mrgb = np.mean(map(float, ratio_data[indx]['nrgb']))
            magb = np.mean(map(float, ratio_data[indx]['nagb']))

            mratio = magb / mrgb
            mratio_err = rsp.utils.count_uncert_ratio(magb, mrgb)

            pct_diff = 1 - (mratio / dratio)
            pct_diff_err = np.abs(pct_diff * (mratio_err / mratio + dratio_err / dratio))


            if 1:
                line += ' & '.join([target.upper(), fmt.format(dratio, dratio_err), fmt.format(mratio, mratio_err), fmt.format(pct_diff, pct_diff_err)])
                line += '\n'
            else:
                line += ' & '.join([fmt.format(mratio, mratio_err), fmt.format(pct_diff, pct_diff_err)])
                line += '\n'
    print line

if __name__ == "__main__":
    main(sys.argv[1:])



# old shit below

def contamination_files(filenames):
    opt_eagb_contam = np.array([])
    opt_rheb_contam = np.array([])
    ir_eagb_contam = np.array([])
    ir_rheb_contam = np.array([])
    opt_ms_contam = np.array([])
    opt_bheb_contam = np.array([])
    ir_ms_contam = np.array([])
    ir_bheb_contam = np.array([])

    if type(filenames) == str:
        filenames = list(filenames)
    for filename in filenames:
        with open(filename, 'r') as fhandle:
            lines = fhandle.readlines()
        # rc contamination 12::13
        rgb_opt = [l for l  in lines if l.startswith('rgb opt')]
        rgb_data = zip(*[t.strip().split()[2:] for t in rgb_opt])
        rgb_data = np.array(rgb_data, dtype=float)
        eagb_in_rgb = rgb_data[5]/rgb_data[7]
        rheb_in_rgb = rgb_data[4]/rgb_data[7]
        opt_eagb_contam = np.append(opt_eagb_contam, np.max(eagb_in_rgb))
        opt_rheb_contam = np.append(opt_rheb_contam, np.max(rheb_in_rgb))
        #print filename, 'opt', np.max(eagb_in_rgb), np.max(rheb_in_rgb)

        opt =  [l for l  in lines if l.startswith('rgb opt') or l.startswith('agb opt')]
        data = zip(*[t.strip().split()[2:] for t in opt])
        data = np.array(data, dtype=float)
        ms_in_opt = data[0]/data[7]
        bheb_in_opt = data[3]/data[7]
        opt_bheb_contam = np.append(opt_bheb_contam, np.max(bheb_in_opt))
        opt_ms_contam = np.append(opt_ms_contam, np.max(ms_in_opt))
        print filename, 'opt', np.max(ms_in_opt), np.max(bheb_in_opt)

        rgb_ir = [l for l  in lines if l.startswith('rgb ir')]
        rgb_data = zip(*[t.strip().split()[2:] for t in rgb_ir])
        rgb_data = np.array(rgb_data, dtype=float)
        eagb_in_rgb = rgb_data[5]/rgb_data[7]
        rheb_in_rgb = rgb_data[4]/rgb_data[7]
        ir_eagb_contam = np.append(ir_eagb_contam, np.max(eagb_in_rgb))
        ir_rheb_contam = np.append(ir_rheb_contam, np.max(rheb_in_rgb))
        #print filename, 'ir', np.max(eagb_in_rgb), np.max(rheb_in_rgb)

        ir =  [l for l  in lines if l.startswith('rgb ir') or l.startswith('agb ir')]
        data = zip(*[t.strip().split()[2:] for t in ir])
        data = np.array(data, dtype=float)
        ms_in_ir = data[0]/data[7]
        bheb_in_ir = data[3]/data[7]
        #print filename, 'ir', np.max(eagb_in_rgb), np.max(rheb_in_rgb)

        ir_bheb_contam = np.append(ir_bheb_contam, np.max(bheb_in_ir))
        ir_ms_contam = np.append(ir_ms_contam, np.max(ms_in_ir))

    print 'opt eagb, rheb', np.max(opt_eagb_contam), np.max(opt_rheb_contam)
    print 'ir eagb, rheb', np.max(ir_eagb_contam), np.max(ir_rheb_contam)
    print 'opt bheb, ms', np.max(opt_bheb_contam), np.max(opt_ms_contam)
    print 'ir bheb, ms', np.max(ir_bheb_contam), np.max(ir_ms_contam)
