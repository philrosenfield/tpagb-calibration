import argparse
import os
import sys
from ..TPAGBparams import EXT, snap_src
import matplotlib.pyplot as plt
import ResolvedStellarPops as rsp
from ..fileio import get_files
from .plotting import outside_labels, emboss
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


    sfhs = [SFH(s) for s in args.name]

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

def default_run():
    #sfh_loc = '/Volumes/tehom/research/TP-AGBcalib/SNAP/varysfh/extpagb/'
    sfh_loc = os.getcwd()
    err_loc = os.path.join(snap_src, 'data/dweisz11_csfr')
    print(os.path.isdir(err_loc))
    targets = ['ngc300-wide1', 'ugc8508', 'ngc4163', 'ngc2403-deep', 'ngc2403-halo-6',
               'ugc4459', 'eso540-030', 'ngc3741', 'ugc5139', 'ugc4305-1',
               'ugc4305-2', 'ugca292', 'kdg73', 'ddo82']

    fig, axs = plt.subplots(ncols=5, nrows=3, figsize=(16, 8))
    axs = outside_labels(axs, fig=fig, xlabel=r'$\rm{Age (Gyr)}$',
                        ylabel=r'$\rm{Cumulative\ SF}$')
    fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0.1)
    line = ''
    for i in range(len(targets)):
        ax = axs[i]
        print targets[i]
        lab = r'$\rm{{{}}}$'.format(targets[i].upper().replace('-','\!-\!'))
        meta_file, = get_files(sfh_loc, '*{}*sfh'.format(targets[i]))
        sfh_file, = get_files(sfh_loc, '*{}*zc.dat'.format(targets[i]))

        sfh = SFH(sfh_file, meta_file=meta_file)
        ax = sfh.plot_csfr(ax=ax)
        
        try:
            err_file, =  get_files(err_loc, '{}*fine'.format(targets[i]))
            esfh = SFH(err_file, meta_file=meta_file)
            esfh.plot_csfr(ax=ax, fill_between_kw={'alpha':0.25}, data=False)
        except:
            import pdb; pdb.set_trace()
            print('systematic uncertainty file {} not found'.format(targets[i]))
        
        d = sfh.param_table()
        line += d['fmt'].format(**d)
        
        ax.text(1, 0.05, lab, ha='right', fontsize=16, **emboss())

    plt.savefig('csfr{}'.format(EXT))
    print(line)


if __name__ == '__main__':
    if '-f' in sys.argv:
        default_run()
    else:
        main(sys.argv[1:])
