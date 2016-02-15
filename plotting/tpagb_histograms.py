import sys, os

from ..fileio import load_lf_file, get_files
from ResolvedStellarPops.galaxies.simgalaxy import SimGalaxy
from .plotting import outside_labels, emboss, tpagb_model_default_color, plot_model
from ..TPAGBparams import snap_src, EXT
from palettable.wesanderson import Darjeeling2_5
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set()
sns.set_context('paper')
plt.style.use('paper')

def plot_labels(dm=0.5):
    ylab = r'$\rm{{Number\ of\ TP\!-\!AGB\ Stars\ /\ {:.1f}\ M_\odot}}$'.format(dm)
    xlab = r'$\rm{Initial\ Mass\ (M_\odot)}$'
    return xlab, ylab


def tpagb_mass(targets, path, oneplot=True, save=True, dm=0.1,
               saved=False):
    axs = [None] * len(targets)

    if oneplot:
        fig, axs = plt.subplots(nrows=len(targets), figsize=(16,16))
        xlab, ylab = plot_labels(dm=dm)
        axs = outside_labels(axs, fig=fig, xlabel=xlab, ylabel=ylab,
                             ylabel_xval=0.05)
        from matplotlib import ticker

    bins = np.arange(0.8, 10. + dm, dm)
    bins, histss, meanhs = load_hists(targets, path, saved=saved, save=save,
                                      key='m_ini', dm=dm, bins=bins)

    for i, target in enumerate(targets):
        ax = axs[i]
        #import pdb; pdb.set_trace()
        minhists = np.min(np.array(histss[i]).T, axis=1)
        maxhists = np.max(np.array(histss[i]).T, axis=1)
        meanhists = np.median(np.array(histss[i]).T, axis=1)
        edges = np.repeat(bins, 2)
        fminhist = np.hstack((0, np.repeat(minhists, 2), 0))
        fmaxhist = np.hstack((0, np.repeat(maxhists, 2), 0))

        ax.fill_between(edges, fminhist, fmaxhist, color='k',
                        alpha='0.2')

        ax.plot(bins[1:], minhists, linestyle='steps', color='k',
                lw=2)
        ax.plot(bins[1:], maxhists, linestyle='steps', color='k',
                lw=2)
        ax.plot(bins[1:], meanhists, linestyle='steps', lw=3, color='k')

        #ax, l =  plot_model(mag2s=histss[i], bins=bins, ax=ax,
        #                    norms=np.ones(len(histss[i])))

        if oneplot:
            ax.text(0.98, 0.9,
                    r'$\rm{{{}}}$'.format(target.upper()).replace('-', '\!-\!'),
                    transform=ax.transAxes, va='top', ha='right')
        else:
            outfile = 'tpagb_{}_hist_{}{}'.format(key, target, EXT)
            plt.close()

    for ax in axs:
        ax.set_yscale('log')
        ax.set_xlim(0.8, 6)

    if oneplot:
        for ax in axs:
            ax.tick_params(labelsize=24)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(prune='both'))
                                                          #nbins=1))
            ax.grid()
        fig.subplots_adjust(hspace=0., left=0.1, bottom=0.1, top=0.95)
    #plt.tight_layout()
    outfile = 'big_tpagb_m_ini_hist{}'.format(EXT)
    plt.savefig(outfile)
    print('wrote {}'.format(outfile))
"""

        #print('{} {:.2f} {:.2f} {:.2f} {:.2f}'.format(target, np.min(mtpagbs),
        #        np.median(mtpagbs), np.mean(mtpagbs), np.max(mtpagbs))),
        #bins = np.arange(0.8, 10.1, 0.1)
        #mhist = np.histogram(mtpagbs, bins=bins)[0]
        #print bins
        #print mhist
        #print np.max(bins[np.nonzero(mhist>100)])
        ##import pdb; pdb.set_trace()
"""

def load_tpagbs(lf_file, sims, path, key='m_ini', gyr=False):
    """
    Load the scaled simulation TP-AGB distribution of some key
    Parameters
    ----------
    lf_file : string
        path to LF formated file
        -- horrific format
    sims : array of strings
        paths to trilegal output files that correspond to the lf_file
    key : str
        column in trilegal output file to read
    gyr : bool
        use Gyr instead of log ages (will over ride 'key' to 'logAge')

    Returns
    -------
    tpagbs : list of arrays
        the scaled tp_agb values of key (sim_agb)
    """
    lfd = load_lf_file(lf_file)
    target = os.path.split(lf_file)[1].split('_')[0]
    # not all sims were included because of normalization factor threshold
    idx = map(int, np.concatenate(lfd['idx']))
    tpagbs = []
    for i in idx:
        # this will not actually work -- need to find the index by filename...
        try:
            sim, = get_files(path, 'out*{}*{:03d}.dat'.format(target, i))
        except:
            print get_files(path, 'out*{}*{:03d}.dat'.format(target, i))
            import pdb; pdb.set_trace()
        sgal = SimGalaxy(sim)
        if gyr:
            key = 'logAge'
            data = 10 ** (sgal.data[key] - 9)
        else:
            data = sgal.data[key]
        tpagbs.append(data[lfd['sim_agb'][i]])
        del sgal
    if key == 'm_ini':
        print('{} {} {:.2f} {:.2f}'.format(os.path.split(lf_file)[1], key,
                                           np.min(np.concatenate(tpagbs)),
                                           np.max(np.concatenate(tpagbs))))
    return tpagbs


def make_hists(tpagbs, dm=0.5, bins=None, norm=True):
    """
    Bin up the list of arrays
    Paramters
    ---------
    tpagbs : list of arrays
        return of load_tpagbs
    dm : float
        bin width (if bins=None)
    bins : array
        bin edges
    norm : bool
        scale histogram by the length of each individual array

    Returns
    -------
    bins : array
        bin edges

    hists : list of arrays
        a list of the histogram of each array

    meanh : array
        median histogram
    """
    if bins is None:
        bins = np.arange(0, 6 + dm, dm)
    if norm:
        hists = [np.histogram(tpagb, bins=bins)[0] / float(len(tpagb))
                 for tpagb in tpagbs]
    else:
        hists = [np.histogram(tpagb, bins=bins)[0] for tpagb in tpagbs]

    meanh = np.median(np.array(hists).T, axis=1)

    return bins, hists, meanh


def save_hists(target, bins, hists, meanh, key='m_ini'):
    """
    save the histograms to a file
    will make two files: one of the histograms, the other for the median
    histogram

    Parameters
    ----------
    target : str
        name of galaxy (for file name)
    bins : array
        histogram bin edges
    hists : list of arrays
        histograms
    meanh : array
        median histogram
    key : str
        column from trilegal (for file name)
    """
    outfile = '{}_tpagb_{}_hists.dat'.format(target, key)
    line = '# target bins hists\n'
    line += ' '.join(map('{:.3f}'.format, bins)) + '\n'
    hline = '\n'.join([' '.join(map('{:d}'.format, hist)) for hist in hists])
    mline = ' '.join(map('{}'.format, meanh)) + '\n'
    with open(outfile, 'w') as out:
        out.write(line)
        out.write(hline)
    outfile = '{}_mean_tpagb_{}_hists.dat'.format(target, key)
    with open(outfile, 'w') as out:
        out.write(line)
        out.write(mline)
    return


def read_hists(targets, path, key='m_ini', mean_only=False):
    """
    Read files produced by save_hists

    Parameters
    ----------
    targets: list of str
        galaxy names (to find histogram files)
    path : str
        location of lf_files and histogram files
    key : str
        column of trilegal output to combine
    mean_only : bool
        only load median histogram
    """
    try:
        mfiles = [get_files(path, '{}_mean_tpagb_{}_hists.dat'.format(t, key))[0]
                  for t in targets]
        hfiles = [get_files(path, '{}_tpagb_{}_hists.dat'.format(t, key))[0]
                  for t in targets]
    except IndexError:
        print('target not found. Perhaps you need to make some {} hists first'.format(key))
        sys.exit(1)
    meanhs = []
    histss = []
    for mfile in mfiles:
        with open(mfile) as m:
            m.readline()
            bins = np.array(m.readline().strip().split(), dtype=float)
            meanhs.append(np.array(m.readline().strip().split(), dtype=float))
    if not mean_only:
        histss = [np.genfromtxt(h, skip_header=2) for h in hfiles]
    return bins, histss, meanhs


def load_hists(targets, path, mean_only=False, saved=False, save=True,
               key='m_ini', dm=0.5, bins=None, norm=False):
    """
    Either call read_hists or make_hists (if the latter, option to save them)

    Parameters
    ----------
    targets: list of str
        galaxy names (to find histogram files)
    path : str
        location of lf_files and histogram files
    key : str
        column of trilegal output to combine
    Passed to read_hists:
        mean_only : bool
            only load median histogram
        saved : bool
            histogram files have been made, call read_hists
        save : bool
            with saved=False, save the new histograms to file
    Passed to make_hists:
        dm : float
            with saved=False bin width
        bins : array
            bin edges
        norm : bool
            scale histogram by the length of each individual array
    Returns
    -------
    bins : array
        bin edges
    histss : list of list of arrays
        all histograms
    meanhs : list of arrays
        all median histograms
    """
    gyr = False
    if key == 'logAge':
        gyr = True
    if saved:
        bins, histss, meanhs = read_hists(targets, path, key=key)
    else:
        meanhs = []
        histss = []
        for target in targets:
            # this is for when some targets are saved, others not.
            print(target)
            outfile = '{}_tpagb_{}_hists.dat'.format(target, key)
            if os.path.isfile(outfile):
                print('not overwriting {}'.format(outfile))
                bins, hists, meanh = read_hists([target], path, key=key)
                hists = hists[0]
                meanh = meanh[0]
            else:
                sims = get_files(path, 'out*{}*.dat'.format(target))
                lf_file, = get_files(path, '*{}*lf.dat'.format(target))
                bins, hists, meanh = make_hists(load_tpagbs(lf_file, sims, path, gyr=gyr),
                                                dm=dm, bins=bins, norm=norm)
                if save:
                    save_hists(target, bins, hists, meanh, key=key)
            histss.append(hists)
            meanhs.append(meanh)

    return bins, histss, meanhs


def stacked_plot(targets, path=None, save=False, saved=False,
                 key='logAge'):
    """
    Create a super cool stacked horizontal bar plot

    Parameters
    ----------
    targets: list of str
        galaxy names (to find histogram files)
    path : str
        location of lf_files and histogram files
    key : str
        column of trilegal output to combine
    Passed to read_hists:
        saved : bool
            histogram files have been made, call read_hists
        save : bool
            with saved=False, save the new histograms to file
    """
    def decorate(ax, bins, targets, colors, key):
        """
        Turn off axis borders, tick marks, add custom legend,
        and set yticks to be target names
        """
        import matplotlib.patches as mpatches
        if key == 'logAge':
            labs = '< {1}, {1}-{2}, {2}-{3}, {3}-{4}, > {4}'.format(*bins)
            labels = map(r'$\rm{{{}\ Gyr}}$'.format, labs.split(','))
        else:
            labs = '{0}-{1}, {1}-{2}, {2}-{3}, {3}-{4}, > {4}'.format(*bins)
            labels = map(r'$\rm{{{}\ M_\odot}}$'.format, labs.split(','))

        patches = [mpatches.Patch(color=colors[i], label=labels[i])
                   for i in range(len(labels))]
        plt.legend(handles=patches, bbox_to_anchor=(.44, -0.05), loc=8, ncol=5)

        ylabs = [r'$\rm{{{}}}$'.format(target.upper()).replace('-', '\!-\!')
                 for target in targets]

        plt.yticks(np.arange(len(targets)), ylabs)
        [spine.set_visible(False) for spine in ax.spines.itervalues()]
        ax.tick_params(labelbottom='off', bottom='off', top='off',
                       left='off', right='off')

    if saved:
        bins, _, meanhs = load_hists(targets, path, saved=True, key=key,
                                     mean_only=True)
    else:
        if key == 'logAge':
            bins = np.array([ 0., 0.3, 1., 1.5, 6.3, 15.])
        else:
            bins = np.array([0.8, 1.2, 1.8, 2.4, 3., 4.])
        bins, _, meanhs = load_hists(targets, path, key=key, save=save,
                                     mean_only=True, bins=bins)

    # bug? Does not make the plots the same width
    if key == 'logAge':
        right = 1.135
        colors = Darjeeling2_5.mpl_colors
    else:
        colors = Darjeeling2_5.mpl_colors[::-1]
        right = 0.98

    bcolors = colors[:]
    bcolors.append(colors[0])  # see iterations with barh below
    bcolors.pop(0)

    fig, ax = plt.subplots(figsize=(14,6))
    fig.subplots_adjust(right=right, left=0.2, bottom=0.05, top=0.98)

    for i, target in enumerate(targets):
        meanh = meanhs[i] / np.sum(meanhs[i])
        ax.barh(i, meanh[0], 0.8, color=bcolors[-1], align='center')
        [ax.barh(i, meanh[j+1], 0.8, left=np.cumsum(meanh)[j],
          align='center', color=bcolors[j]) for j in range(len(meanh))[:-1]]

    decorate(ax, bins, targets, colors, key)
    outfile = 'tpagb_{}_hists{}'.format(key, EXT)
    plt.savefig(outfile)
    plt.close()
    print('wrote {}'.format(outfile))

def default_run():
    path = snap_src + '/varysfh/extpagb/final_hope/keep/all_run/caf09_v1.2s_m36_s12d_ns_nas'
    targets = ['ngc300-wide1',
               'ugc8508',
               'ngc4163',
               'ngc2403-deep',
               'ngc2403-halo-6',
               'ugc4459',
               #'eso540-030',
               'ugc4305-1',
               'ugc4305-2',
               'ngc3741',
               'ugc5139',
               'kdg73',
               'ugca292'][::-1]
    #How to include more galaxies ... set saved=False.
    #new = ['ngc300-wide1', 'ugc4305-2', 'ugca292']
    #new = ['ugc8508']
    #stacked_plot(targets, path=path, saved=False, save=False, key='m_ini')
    #stacked_plot(targets, path=path, saved=False, save=True, key='m_ini')

    #stacked_plot(targets, path=path, saved=True, key='m_ini')
    #stacked_plot(targets, path=path, saved=True, key='logAge')
    tpagb_mass(targets[::-1], path, save=False, saved=True)

if __name__ == "__main__":
    default_run()
