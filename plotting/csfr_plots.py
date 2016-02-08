import argparse
import os
import sys
from ..TPAGBparams import EXT, snap_src
import matplotlib.pyplot as plt
import ResolvedStellarPops as rsp
from ..fileio import get_files
from .plotting import outside_labels, emboss
from dweisz.match.scripts.sfh import SFH, quadriture
from dweisz.match.scripts.fileio import read_binned_sfh, parse_pipeline
from dweisz.match.scripts.utils import convertz
import numpy as np

def param_table(sfh, angst=True, agesplit=[1e9, 3e9], target='',
                filters=['','']):
    try:
        d = {'bestfit': sfh.bestfit, 'Av': sfh.Av, 'dmod': sfh.dmod}
        d['dist'] = 10 ** (sfh.dmod / 5. + 1) / 1e6
    except:
        print('No bestfit info')
        d = {'bestfit': np.nan, 'Av': np.nan, 'dmod': np.nan}
        d['dist'] = 0.

    d['header'] = \
        (r'Galaxy & Optical Filters & A$_V$ & $(m\!-\!M)_0$ & $D$ &'
         r'$\% \frac{{\rm{{SF}}}}{{\rm{{SF_{{TOT}}}}}}$ &'
         r'$\langle \mbox{{[Fe/H]}} \rangle$ &'
         r'$\% \frac{{\rm{{SF}}}}{{\rm{{SF_{{TOT}}}}}}$ &'
         r'$\langle \mbox{{[Fe/H]}} \rangle$ & $bestfit$ \\ & & & & '
         r'\multicolumn{{2}}{{c}}{{$<{0}\rm{{Gyr}}$}} & '
         r'\multicolumn{{2}}{{c}}{{${0}-{1}\rm{{Gyr}}$}} & \\ \hline'
         '\n'.format(*agesplit))

    d['target'] = target
    if angst:
        try:
            d['target'], filters = parse_pipeline(sfh.name)
        except:
            pass

    d['filters'] = ','.join(filters)

    fyoung, fyoung_errp, fyoung_errm = sfh.mass_fraction(0, agesplit[0])
    finter, finter_errp, finter_errm = sfh.mass_fraction(agesplit[0], agesplit[1])

    # logZ = 0 if there is no SF, that will add error to mean Fe/H
    iyoung = sfh.nearest_age(agesplit[0], i=False)
    iinter = sfh.nearest_age(agesplit[1], i=False)

    iyoungs, = np.nonzero(sfh.data.mh[:iyoung + 1] != 0)
    iinters, = np.nonzero(sfh.data.mh[:iinter + 1] != 0)
    iinters = list(set(iinters) - set(iyoungs))

    feh_young = convertz(z=0.02 * 10 ** np.mean(sfh.data.mh[iyoungs]))[-2]
    feh_inter = convertz(z=0.02 * 10 ** np.mean(sfh.data.mh[iinters]))[-2]
    feh_young_errp = convertz(z=0.02 * 10 ** quadriture(sfh.data.mh_errp[iyoungs]))[-2]
    feh_young_errm = convertz(z=0.02 * 10 ** quadriture(sfh.data.mh_errm[iyoungs]))[-2]
    feh_inter_errp = convertz(z=0.02 * 10 ** quadriture(sfh.data.mh_errp[iinters]))[-2]
    feh_inter_errm = convertz(z=0.02 * 10 ** quadriture(sfh.data.mh_errm[iinters]))[-2]

    maf = '${0: .2f}^{{+{1: .2f}}}_{{-{2: .2f}}}$'

    d['fyoung'], d['finter'] = [maf.format(v, p, m)
                                for v,p,m in zip([fyoung, finter],
                                                 [fyoung_errp, finter_errp],
                                                 [fyoung_errm, finter_errm])]
    d['feh_young'], d['feh_inter'] = [maf.format(v, p, m)
                                      for v,p,m in zip([feh_young, feh_inter],
                                                       [feh_young_errp, feh_inter_errp],
                                                       [feh_young_errm, feh_inter_errm])]

    line = ['{target}', '{filters}', '{Av: .2f}', '{dmod: .2f}', '{dist: .2f}',
            '{fyoung}', '{feh_young}','{finter}',
            '{feh_inter}']#, '{bestfit: .1f}']

    d['fmt'] = '%s \\\\ \n' % (' & '.join(line))
    return d


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
    targets = ['ngc300-wide1', 'ugc8508', 'ngc4163', 'ngc2403-deep',
               'ngc2403-halo-6', 'ugc4459', 'ngc3741', 'ugc5139', 'ugc4305-1',
               'ugc4305-2', 'kdg73', 'ugca292']

    fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(16, 8))
    axs = outside_labels(axs, fig=fig, xlabel=r'$\rm{Age\ (Gyr)}$',
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
            err_file, = get_files(err_loc, '{}*fine'.format(targets[i]))
            esfh = SFH(err_file, meta_file=meta_file)
            esfh.plot_csfr(ax=ax, fill_between_kw={'alpha':0.25}, data=False)
        except:
            #import pdb; pdb.set_trace()
            print('systematic uncertainty file {} not found'.format(targets[i]))

        d = param_table(sfh)
        line += d['fmt'].format(**d)
        if 'HALO' in lab or 'UGCA' in lab:
            ax.text(13.95, 0.95, lab, ha='left', va='top', fontsize=16, **emboss())
        else:
            ax.text(0.05, 0.05, lab, ha='right', fontsize=16, **emboss())

    plt.savefig('csfr{}'.format(EXT))
    print(line)


if __name__ == '__main__':
    if '-f' in sys.argv:
        default_run()
    else:
        main(sys.argv[1:])
