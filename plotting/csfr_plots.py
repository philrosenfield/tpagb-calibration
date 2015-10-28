import argparse
import os
import sys

import matplotlib.pyplot as plt
from ..fileio import get_files
from dweisz.match.scripts.sfh import SFH as MatchSFH

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
        plt.savefig('all_csfr.png')



def default_run():
    sfh_loc = '/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/extpagb/'
    sfh_files = get_files(sfh_loc, '*sfh')
    targets = [os.path.split(l)[1].split('_')[0] for l in sfh_files]
    hmc_files = [get_files(sfh_loc, '{}*.mcmc.zc'.format(t))[0] for t in targets]
    sfhs = [MatchSFH(sfh_files[i], hmc_files[i]) for i in range(len(targets))]
    print sfh_files
    [s.plot_csfr() for s in sfhs]

if __name__ == '__main__':
    if '-f' in sys.argv:
        default_run()
    else:
        main(sys.argv[1:])
