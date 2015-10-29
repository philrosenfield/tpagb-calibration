import argparse
import os
import sys
from ..TPAGBparams import EXT
import matplotlib.pyplot as plt
from ..fileio import get_files
from dweisz.match.scripts.sfh import SFH

plt.style.use('presentation')

def main(argv):
    parser = argparse.ArgumentParser(description="make csfr plots")

    parser.add_argument('name', nargs='*', type=str, help='match hmc file(s)')

    parser.add_argument('-k', '--file_origin', type=str, default='match-hmc',
                        help='type of match sfh file')

    parser.add_argument('-a', '--one_plot', action='store_true',
                        help='all files on one plot')

    parser.add_argument('-e', '--errors', action='store_true',
                        help='add errors')

    args = parser.parse_args(argv)

    sfhs = [rsp.match.utils.MatchSFH(s) for s in args.name]
    targets = [s.name.split('_')[0] for s in sfhs]

    if args.one_plot:
        fig, ax = plt.subplots()
        colors = rsp.graphics.discrete_colors(len(sfhs), cmap=plt.cm.RdYlBu)
    else:
        ax = None
        colors = ['k'] * len(sfhs)

    for i, sfh in enumerate(sfhs):
        ax = sfh.plot_csfr(ax=ax, errors=args.errors,
                           plt_kw={'color': colors[i],
                                   'label': r'${}$'.format(targets[i].upper())},
                           fill_between_kw={'color': colors[i]})
        if not args.one_plot:
            ax = None
    if args.one_plot:
        ax.set_xlabel('$\\rm{Time\ (Gyr)}$')
        ax.set_ylabel('$\\rm{Culmulative\ SF}$')
        plt.legend(loc=0, frameon=False)
        plt.savefig('all_csfr{}'.format(EXT))

def emboss(fg='w', lw=3):
    from matplotlib.patheffects import withStroke
    myeffect = withStroke(foreground=fg, linewidth=lw, alpha=0.5)
    return dict(path_effects=[myeffect])

def outside_labels(axs, fig=None, xlabel=None, ylabel=None):
    """
    make an ndarray of axes only have lables on the outside of the grid
    also add common x and y label if fig, xlabel, ylabel passed.
    Returns
        np.ravel(axs)
    """
    [ax.tick_params(labelleft=False, direction='in', which='both')
     for ax in np.ravel(axs)]
    [ax.tick_params(labeltop=True, labelbottom=False) for ax in axs[0, :]]
    [ax.tick_params(labelleft=True) for ax in [axs[0, 0], axs[1, 0]]]
    [ax.tick_params(labelright=True) for ax in [axs[0, -1], axs[1, -1]]]
    axs = np.ravel(axs)

    if xlabel is not None:
        fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=24)

    if ylabel is not None:
        fig.text(0.06, 0.5, ylabel, ha='center', va='center', fontsize=24,
                 rotation='vertical')
    return axs


def default_run():
    sfh_loc = '/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/extpagb/'
    targets = ['ugc8508', 'ngc4163', 'ngc2403-deep', 'ngc2403-halo-6',
               'ugc4459', 'eso540-030', 'ngc3741', 'ugc5139', 'ugc4305-1', 'kdg73']

    fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(16, 8))
    axs = outside_labels(axs, fig=fig, xlabel=r'$\rm{Age (Gyr)}$',
                        ylabel=r'$\rm{Cumulative\ SF}$')
    fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0.1)

    for i in range(len(targets)):
        ax = axs[i]
        lab = r'$\rm{{{}}}$'.format(targets[i].upper().replace('-','\!-\!'))
        sfh_file, = get_files(sfh_loc, '*{}*sfh'.format(targets[i]))
        hmc_file, = get_files(sfh_loc, '*{}*.mcmc.zc'.format(targets[i]))

        sfh = SFH(sfh_file, hmc_file)
        ax = sfh.plot_csfr(ax=ax)
        ax.text(1, 0.05, lab, ha='right', fontsize=16, **emboss())

    plt.savefig('csfr{}'.format(EXT))


if __name__ == '__main__':
    if '-f' in sys.argv:
        default_run()
    else:
        main(sys.argv[1:])
