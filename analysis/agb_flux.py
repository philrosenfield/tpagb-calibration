import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from ResolvedStellarPops.fileio import readfile, get_files
from ResolvedStellarPops.utils import count_uncert_ratio
from ResolvedStellarPops.galaxies.galaxy import Galaxy

from dweisz.match.scripts.sfh import SFH as MatchSFH
from ..fileio import load_lf_file, load_observation
from .analyze import get_itpagb

plt.style.use('presentation')

def main(argv):
    parser = argparse.ArgumentParser(description="Plot LFs against galaxy data")

    parser.add_argument('-e', '--hmcfiles', type=str, nargs='*',
                        help='HMC files')

    parser.add_argument('-f', '--force', action='store_true',
                        help='over ride all parameters and run default files')

    parser.add_argument('sfhfiles', type=str, nargs='*',
                        help='CalcSFH output files')

    parser.add_argument('narratio_files', type=str, nargs='*',
                        help='model narratio files')

    parser.add_argument('lf_files', type=str, nargs='*',
                        help='model LFs files')

    parser.add_argument('observations', type=str, nargs='*',
                        help='data files to compare to')

    args = parser.parse_args(argv)

    make_plot(args.narratio_files, args.sfhfiles, args.lffiles, args.observations, hmcfiles=args.hmcfiles)

if __name__ == '__main__':
    if '-f' in sys.argv[1:]:
        default_run()
    else:
        main(sys.argv[1:])


def default_run():
    import os
    lf_loc = '/Volumes/tehom/keep/all_run/caf09_v1.2s_m36_s12d_ns_nas'
    obs_loc = '/Volumes/tehom/andromeda/research/TP-AGBcalib/SNAP/data/opt_ir_matched_v2/copy/'
    sfh_loc = '/Volumes/tehom/andromeda/research/TP-AGBcalib/SNAP/varysfh/extpagb/'

    lf_files = get_files(lf_loc, '*lf.dat')
    targets = [os.path.split(l)[1].split('_')[0] for l in lf_files]
    observations = [get_files(obs_loc, '{}*fits'.format(t))[0] for t in targets]
    narratio_files = [get_files(lf_loc, '{}*nar*dat'.format(t))[0] for t in targets]
    sfh_files = [get_files(sfh_loc, '{}*sfh'.format(t))[0] for t in targets]
    hmc_files = [get_files(sfh_loc, '{}*.mcmc.zc'.format(t))[0] for t in targets]

    make_plot(narratio_files, sfh_files, lf_files, observations, hmcfiles=hmcfiles)


def add_inset(ax0, extent, xlims, ylims):
    '''
    add an inset axes and a rectangle on the main plot

    Parameters
    ----------
    ax0 : parent axis instance
        main axes (to add rectangular patch)
    extent : list
        extent is passed to plt.axes from mpl.axes.__doc__:
           ``axes(rect, axisbg='w')`` where *rect* = [left, bottom, width,
           height] in normalized (0, 1) units.  *axisbg* is the background
           color for the axis, default white.
    xlims, ylims : list, list
        axes limits of the inset

    Returns
    -------
        ax : inset axes instance
    '''
    ax = plt.axes(extent)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    rect = matplotlib.patches.Rectangle((xlims[0], ylims[0]),
                                        np.diff(xlims), np.diff(ylims),
                                        fill=False, color='k')
    ax0.add_patch(rect)
    return ax



def ntpagb_model_data(narratiofile, sfhfile, hmcfile=None):
    """

    """
    ratiodata = readfile(narratiofile, string_column=[0, 1, 2])
    inds, = np.nonzero(ratiodata['target'] != 'data')
    if len(inds) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean_nagb = np.mean(ratiodata['nagb'][1:])
    agbm2d = mean_nagb / ratiodata['nagb'][0]

    sfh = MatchSFH(sfhfile, hmc_file=hmcfile)
    massfrac, massfrac_perr, massfrac_merr = sfh.mass_fraction(0, 2e9)

    agbm2d_err = count_uncert_ratio(np.mean(ratiodata['nagb'][1:]),
                                            ratiodata['nagb'][0])
    return massfrac, massfrac_perr, massfrac_merr, agbm2d, agbm2d_err

def make_plot(narratio_files, sfhfiles, lffiles, observations, hmcfiles=None,
              inset=False):
    def _plot(ax, targets, f, massfrac, m2d, m2d_err, massfrac_perr,
              massfrac_merr):
        ax = make_melbourne_plot(ax=ax, targets=targets, flux=f)
        color = '#30a2da'
        ax.plot(massfrac, m2d, 'o', color=color, label='R14', ms=10, zorder=1000)
        ax.errorbar(massfrac, m2d, yerr=np.array(m2d_err) / 2,
                    xerr=[massfrac_perr, massfrac_merr],
                    fmt='none', ecolor=color, zorder=1000)
        ax.axhline(np.nanmean(m2d), ls='--', color=color)
        ax.axhline(1., color='k', lw=2)
        return ax

    if hmcfiles is None:
        hmcfiles = [None] * len(sfhfiles)

    massfrac, massfrac_perr, massfrac_merr, agbm2d, agbm2d_err = \
        zip(*[ntpagb_model_data(narratio_files[i], sfhfiles[i], hmcfile=hmcfiles[i])
              for i in range(len(sfhfiles))])

    fagbm2d, fagbm2d_err = zip(*[ftpagb_model_data(lffiles[i], observations[i])
                            for i in range(len(observations))])

    targets = [os.path.split(s)[1].split('_')[0] for s in sfhfiles]

    flux = [False, True]
    ytitles = [latexify('\# TP-AGB (model) / \# TP-AGB (data)'),
               latexify('TP-AGB Flux (model) / TP-AGB Flux (data)')]

    for m2d, m2d_err, f in zip([agbm2d, fagbm2d], [agbm2d_err, fagbm2d_err], flux):
        fig, ax = plt.subplots()
        top = 0.95
        fig.subplots_adjust(left=0.12, bottom=0.15, top=top)
        _plot(ax, targets, f, massfrac, m2d, m2d_err, massfrac_perr,
              massfrac_merr)
        plt.legend(loc='best')
        if f:
            ax.set_ylabel(ytitles[1])
            title = 'agbflux_m2d.png'
        else:
            ax.set_ylabel(ytitles[0])
            title = 'magb_m2d.png'

        if inset:
            yext = [0.6, 0.90]
            ylim = [0.3, 2]
            if f:
                ylim = [0.3, 2]
            ax1 = add_inset(ax, [0.18, 0.65, 0.3, 0.28], [-0.005, 0.041], ylim)
            _plot(ax1, targets, f, massfrac, m2d, m2d_err, massfrac_perr,
                  massfrac_merr)

        ax.set_xlabel(latexify('Mass Fraction of Young Stars (<2 Gyr)'))
        plt.savefig(title)
    return ax

def make_melbourne_plot(tablefile='default', ax=None, targets=None, flux=False):
    if tablefile=='default':
        tablefile = '/Volumes/tehom/andromeda/research/TP-AGBcalib/tables/melbourne2012_tab.dat'
    meltab = readfile(tablefile, string_column=0)

    if targets is None:
        inds = np.arange(len(meltab['Galaxy']))
    else:
        inds = np.concatenate([[i for i, g in enumerate(meltab['Galaxy']) if t.upper() in g] for t in targets])
        inds = map(int, inds)

    if ax is None:
        fig, ax = plt.subplots()

    if flux:
        n10 = 'fAGB10_ftot'
        n08 = 'fAGB08_ftot'
    else:
        n10 = 'NAGB10'
        n08 = 'NAGB08'

    kws = [{'color': 'gray', 'ms': 10, 'marker': 'd', 'mfc': 'w', 'label': 'Padova08'},
           {'color': 'gray', 'ms': 10, 'marker': 's', 'mfc': 'w', 'label': 'Padova10'}]
    for i, n in enumerate([n08, n10]):
        x = meltab['fmass_y_2gr'][inds]
        y = meltab[n][inds]
        xerr = meltab['fmass_y_2gr_err'][inds] / 2
        yerr = meltab['{}_err'.format(n)][inds] / 2
        ax.plot(x, y, linestyle='none', **kws[i])
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='none', ecolor=kws[i]['color'])
        ax.axhline(np.mean(y), ls='--', color=kws[i]['color'])
    return ax

def ftpagb_model_data(lffile, observation):
    lfd = load_lf_file(lffile)
    #print lffile
    try:
        mags = lfd['F160W']
    except:
        mags = lfd['F160W_cor']

    mtpagbs = lfd['sim_agb']

    gal = Galaxy(observation)
    mtrgb, Av, dmod = gal.trgb_av_dmod('F160W')

    try:
        color = gal.data['MAG2_ACS'] - gal.data['MAG4_IR']
    except:
        color = gal.data['MAG2_WFPC2'] - gal.data['MAG4_IR']

    mag2 = gal.data['MAG4_IR']
    dtpagb = get_itpagb(lffile.split('_')[0], color, mag2, 'F160W', dmod=dmod, Av=Av, mtrgb=mtrgb)
    dagbflux = np.sum(10 ** (-.4 * mag2[dtpagb]))
    magbfluxs = [np.sum(10 ** (-.4 * mags[i][mtpagbs[i]])) for i in range(len(mags))]

    #print 'data', dagbflux
    #for i in range(len(mtpagbs)):
    #    print lffile.split('_')[0], magbfluxs[i]

    return np.mean(magbfluxs / dagbflux), np.std(magbfluxs / dagbflux)

def latexify(string):
    return r'$\rm{{{}}}$'.format(string.replace(' ', '\ '))
