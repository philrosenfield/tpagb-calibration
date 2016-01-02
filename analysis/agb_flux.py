import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from ..TPAGBparams import EXT
from ResolvedStellarPops.fileio import readfile, get_files
from ResolvedStellarPops.utils import count_uncert_ratio
from ResolvedStellarPops.galaxies.galaxy import Galaxy

from dweisz.match.scripts.sfh import SFH as MatchSFH
from ..fileio import load_lf_file, load_observation
from .analyze import get_itpagb

plt.style.use('presentation')


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
    import  matplotlib
    ax = plt.axes(extent)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    rect = matplotlib.patches.Rectangle((xlims[0], ylims[0]),
                                        np.diff(xlims), np.diff(ylims),
                                        fill=False, color='k')
    ax0.add_patch(rect)
    return ax


def ntpagb_model_data(narratiofile, sfhfile, metafile=None):
    """
    Compare the number of TP-AGB stars in the model to the data.
    Parameters
    ----------
    narratiofile : formatted files containing the number ratio of TP-AGB stars
    and RGB stars etc. See normalize.py

    sfhfile : the output of calcsfh (to calculate the mass in each sf age bin)
    metafile : if you want Hybrid MC errors reported
    Returns
    -------
    massfrac, massfrac_perr, massfrac_merr : floats
        the mass fraction between 0, 2e9 years and associated uncertainties

    agbm2d, agbm2d_err : the number of TP-AGB stars compared to the data
        and (even) Poisson uncertainties.

    NB
    The first line in the narratiofile is a measurement of the data.
    Set up to take the mass fraction at 0 (youngest age calculated)
    and 2e9. If adjusting, see sfh.mass_fraction.__doc__, currently,
    units can by yr or log yr.
    """
    ratiodata = readfile(narratiofile, string_column=[0, 1, 2])
    inds, = np.nonzero(ratiodata['target'] != 'data')
    if len(inds) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean_nagb = np.mean(ratiodata['nagb'][1:])
    agbm2d = mean_nagb / ratiodata['nagb'][0]

    sfh = MatchSFH(sfhfile, meta_file=metafile)
    massfrac, massfrac_perr, massfrac_merr = sfh.mass_fraction(0, 2e9)

    agbm2d_err = count_uncert_ratio(np.mean(ratiodata['nagb'][1:]),
                                            ratiodata['nagb'][0])
    return massfrac, massfrac_perr, massfrac_merr, agbm2d, agbm2d_err


def make_plot(narratio_files, sfhfiles, lffiles, observations, metafiles=None,
              inset=False):
    """
    Make two plots to compare with Figure 7 of Melbourne+ 2012 Apj 748, 47
    Parameters
    ----------
    narratio_files : list of str
        Formatted files from normalize.py with the AGB/RGB ratio data
    sfhfiles : list of str
        outputs of calcsfh
    lffiles : list of str
        Formatted files from normalize.py with the scaled LFs
    observations : list of str
        Fits files to compare with (apples to apples, should use 4 filter files)
    metafiles : list of str
        Hybrid MonteCarlo outputs associated with each SFH file.

    NB
    Dude, I'm trying to publish. There are NO checks to make sure each target
    is consistent across each of the 5 f*ing file types. No check to see if
    the list lentghs are the same. In the time I am writing this doc string,
    I could have written those tests, but at some point, and that point is now,
    you gotta choose, documenting or coding? (See the nice -f hack around argparse)
    """
    def _plot(ax, targets, f, massfrac, m2d, m2d_err, massfrac_perr,
              massfrac_merr, wmean=None):
        if wmean is None:
            wmean = np.nanmean(m2d)
        ax = make_melbourne_plot(ax=ax, targets=None, flux=f)
        color = '#30a2da'
        ax.plot(massfrac, m2d, 'o', color=color, label='R14', ms=10, zorder=1000)
        ax.errorbar(massfrac, m2d, yerr=np.array(m2d_err) / 2,
                    xerr=[massfrac_perr, massfrac_merr],
                    fmt='none', ecolor=color, zorder=1000)
        ax.axhline(wmean, ls='--', color=color)
        ax.axhline(1., color='k', lw=2)
        print(wmean)
        return ax

    if metafiles is None:
        metafiles = [None] * len(sfhfiles)

    massfrac, massfrac_perr, massfrac_merr, agbm2d, agbm2d_err = \
        zip(*[ntpagb_model_data(narratio_files[i], sfhfiles[i], metafile=metafiles[i])
              for i in range(len(sfhfiles))])

    fagbm2d, fagbm2d_err = zip(*[ftpagb_model_data(lffiles[i], observations[i])
                            for i in range(len(observations))])

    targets = [os.path.split(s)[1].split('_')[0] for s in sfhfiles]

    for i in range(len(sfhfiles)):
        print targets[i], massfrac[i]

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
            title = 'agbflux_m2d{}'.format(EXT)
            ylim = [0, 11]
        else:
            ax.set_ylabel(ytitles[0])
            title = 'magb_m2d{}'.format(EXT)
            ylim = [0, 7]
        if inset:
            yext = [0.6, 0.90]
            iylim = [0.3, 2]
            if f:
                iylim = [0.3, 2]
            ax1 = add_inset(ax, [0.18, 0.65, 0.3, 0.28], [-0.005, 0.041], iylim)
            _plot(ax1, targets, f, massfrac, m2d, m2d_err, massfrac_perr,
                  massfrac_merr)

        ax.set_xlabel(latexify('Mass Fraction of Young Stars (<2 Gyr)'))
        ax.set_xlim(-0.01, 0.45)
        ax.set_ylim(ylim)
        plt.savefig(title)
    return ax

def make_melbourne_plot(tablefile='default', ax=None, targets=None, flux=False):
    """
    Toss Melbourne's data on the plot.
    """
    if tablefile=='default':
        tablefile = '/Volumes/tehom/research/TP-AGBcalib/tables/melbourne2012_tab.dat'
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
        wmeans = {n10: 2.3, n08: 2.5}
    else:
        n10 = 'NAGB10'
        n08 = 'NAGB08'
        wmeans = {n10: 1.5, n08: 2.2}

    kws = [{'color': 'gray', 'ms': 10, 'marker': 'd', 'mfc': 'w', 'label': 'Padova08'},
           {'color': 'gray', 'ms': 10, 'marker': 's', 'mfc': 'w', 'label': 'Padova10'}]
    for i, n in enumerate([n08, n10]):
        x = meltab['fmass_y_2gr'][inds]
        y = meltab[n][inds]
        xerr = meltab['fmass_y_2gr_err'][inds] / 2
        yerr = meltab['{}_err'.format(n)][inds] / 2
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='none', ecolor=kws[i]['color'])
        ax.plot(x, y, linestyle='none', **kws[i])
        ax.axhline(wmeans[n], ls='--', color=kws[i]['color'])
        print(np.mean(y))
    return ax

def ftpagb_model_data(lffile, observation):
    """
    Calculate the TP-AGB flux from the data and model
    """
    lfd = load_lf_file(lffile)
    #print lffile
    try:
        mags = lfd['F160W_cor']
    except:
        mags = lfd['F160W']

    mtpagbs = lfd['sim_agb']

    gal = Galaxy(observation)
    mtrgb, Av, dmod = gal.trgb_av_dmod('F160W')
    trgb_color = gal.trgb_av_dmod('F814W')[0] - mtrgb

    try:
        color = gal.data['MAG2_ACS'] - gal.data['MAG4_IR']
    except:
        color = gal.data['MAG2_WFPC2'] - gal.data['MAG4_IR']

    mag2 = gal.data['MAG4_IR']
    dtpagb = get_itpagb(lffile.split('_')[0], color, mag2, 'F160W', dmod=dmod, Av=Av, mtrgb=mtrgb,
                        off=trgb_color)
    dagbflux = np.sum(10 ** (-.4 * mag2[dtpagb]))
    magbfluxs = [np.sum(10 ** (-.4 * mags[i][mtpagbs[i]])) for i in range(len(mags))]

    #print 'data', dagbflux
    #for i in range(len(mtpagbs)):
    #    print lffile.split('_')[0], magbfluxs[i]

    return np.mean(magbfluxs / dagbflux), 1.5 * np.std(magbfluxs / dagbflux)

def latexify(string):
    """.format is so pretty with latex and curly brackets"""
    return r'$\rm{{{}}}$'.format(string.replace(' ', '\ '))

def default_run():
    """Will you stop writing code and publish for f*s sake. """
    import os
    lf_loc = '/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/extpagb/final_hope/keep/all_run/caf09_v1.2s_m36_s12d_ns_nas'
    obs_loc = '/Volumes/tehom/research/TP-AGBcalib/SNAP/data/opt_ir_matched_v2/copy/'
    sfh_loc = '/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/extpagb/final_hope/'

    lf_files = get_files(lf_loc, '*lf.dat')
    targets = [os.path.split(l)[1].split('_')[0] for l in lf_files]
    observations = [get_files(obs_loc, '{}*fits'.format(t))[0] for t in targets]
    narratio_files = [get_files(lf_loc, '{}*nar*dat'.format(t))[0] for t in targets]
    meta_files = [get_files(sfh_loc, '{}*sfh'.format(t))[0] for t in targets]
    sfh_files = [get_files(sfh_loc, '{}*.mcmc.zc.dat'.format(t))[0] for t in targets]

    make_plot(narratio_files, sfh_files, lf_files, observations, metafiles=meta_files,
              inset=False)


def main(argv):
    parser = argparse.ArgumentParser(description="Plot LFs against galaxy data")

    parser.add_argument('-e', '--metafiles', type=str, nargs='*',
                        help='files with dmod, av, uncertainties files')

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

    make_plot(args.narratio_files, args.sfhfiles, args.lffiles, args.observations, metafiles=args.metafiles)

if __name__ == '__main__':
    if '-f' in sys.argv[1:]:
        # FUCK YEAH LETS DO THIS
        default_run()
    else:
        main(sys.argv[1:])
