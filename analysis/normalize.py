"""
Utilities to analyze trilegal output catalogs of TPAGB models
"""
import argparse
import logging
import numpy as np
import os
import sys

from ..TPAGBparams import EXT
from .analyze import get_itpagb, parse_regions, get_trgb
from ..pop_synth.stellar_pops import (normalize_simulation, rgb_agb_regions,
                                      exclude_gate_inds)
from ..pop_synth import SimGalaxy
from ..fileio import load_observation
from ..plotting import plotting
from ..utils import count_uncert_ratio, parse_pipeline

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from argparse import RawTextHelpFormatter

def do_normalization(yfilter=None, filter1=None, filter2=None, ast_cor=False,
                     sgal=None, tricat=None, nrgbs=None, regions_kw={}, Av=0.,
                     match_param=None, norm=None, tol=0.7):
    '''Do the normalization: call rgb_agb_regions and normalize_simulations.'''

    if sgal is None:
        sgal = SimGalaxy(tricat)
        if sgal is None:
            return None, {}, []

    if match_param is not None:
        _, (f1, f2) = parse_pipeline(match_param)
        logger.info('using exclude gates on simulation')
        m1 = sgal.data[f1]
        m2 = sgal.data[f2]
        inds = exclude_gate_inds(m1, m2, match_param=match_param)
    else:
        inds = np.arange(len(sgal.data[filter2]))

    ymag = sgal.data[filter2][inds]
    if yfilter == 'V':
        ymag = sgal.data[filter1][inds]

    regions_kw['color'] = sgal.data[filter1][inds] - sgal.data[filter2][inds]

    # select rgb and agb regions
    sgal_rgb, sgal_agb = rgb_agb_regions(ymag, **regions_kw)

    if 'IR' in filter2 or '160' in filter2 and not '110' in filter1:
        target = 'sim'
        ir_mtrgb = get_trgb(target, filter2='F160W', filter1=None)
        opt_mtrgb = get_trgb(target, filter2='F814W')
        trgb_color = opt_mtrgb - ir_mtrgb
        sgal_agb = get_itpagb(target, regions_kw['color'], ymag, filter2,
                              off=trgb_color)

    # normalization
    norm, idx_norm, sim_rgb, sim_agb = normalize_simulation(ymag, nrgbs,
                                                            sgal_rgb, sgal_agb,
                                                            norm=norm)

    if norm <= tol:
        norm_dict = {'norm': norm, 'sim_rgb': sim_rgb, 'sim_agb': sim_agb,
                     'sgal_rgb': sgal_rgb, 'sgal_agb': sgal_agb,
                     'idx_norm': idx_norm}
    else:
        norm_dict = {}
        inds = np.array([])
        logger.warning('Normalization > {}. Will not include {}'.format(tol, sgal.name))
    return sgal, norm_dict, inds


def tpagb_lf(sgal, narratio_dict, inds, filt1, filt2, lf_line=''):
    """format a narratio_dict for a line in the LF output file
    Parameters
    ----------
    sgal : SimGalaxy

    narratio_dict : dict
        see narratio.__doc__
    inds : array
        inds of sgal.data, full array unless exclude gates were used
    filt1 : str
        name of filter1
    filt2 : str
        name of filter2
    lf_line : str
        if adding many sgals together
        format is stupid and makes for slow reading.
        each row is:
        filt1 : mag1 full simulation length = inds
        filt2 : mag2 full simulation length = inds
        logAge : logAge full simulation length = inds
        [M/H] : [M/H] full simulation length = inds
        sim_rgb : index array of sgal from narratio_dict
        sim_agb : index array of sgal from narratio_dict
        sgal_rgb : index array of sgal from narratio_dict
        sgal_agb : index array of sgal from narratio_dict
        idx_norm : index array of sgal from narratio_dict
        norm : float, from narratio_dict
        idx : int or str, simulation number or "bestsfr" (last value before .dat in filename)
    """

    header = '# {} {} '.format(filt1, filt2)
    header += 'logAge [M/H] sim_rgb sim_agb sgal_rgb sgal_agb idx_norm norm idx'

    if len(lf_line) == 0:
        lf_line = header
    idx = sgal.name.split('_')[-1].replace('.dat', '')
    lf_line += '\n' + '\n'.join([' '.join(['%g' % m for m in sgal.data[filt1][inds]]),
                                 ' '.join(['%g' % m for m in sgal.data[filt2][inds]]),
                                 ' '.join(['%g' % m for m in sgal.data['logAge'][inds]]),
                                 ' '.join(['%g' % m for m in sgal.data['[M/H]'][inds]]),
                                 ' '.join(['%i' % m for m in narratio_dict['sim_rgb']]),
                                 ' '.join(['%i' % m for m in narratio_dict['sim_agb']]),
                                 ' '.join(['%i' % m for m in narratio_dict['sgal_rgb']]),
                                 ' '.join(['%i' % m for m in narratio_dict['sgal_agb']]),
                                 ' '.join(['%i' % m for m in narratio_dict['idx_norm']]),
                                 '%.4f' % narratio_dict['norm'],
                                 '%s' % idx])
    return lf_line


def narratio(target, nrgb, nagb, filt1, filt2, narratio_line=''):
    """format numbers of stars for the narratio table"""
    # N agb/rgb ratio file
    narratio_fmt = 'galaxy %(filter1)s %(filter2)s %(nrgb)i %(nagb)i %(ar_ratio).3f %(ar_ratio_err).3f\n'
    try:
       ar_ratio = nagb / nrgb
    except ZeroDivisionError:
        ar_ratio = np.nan
    out_dict = {'filter1': filt1,
                'filter2': filt2,
                'ar_ratio': ar_ratio,
                'ar_ratio_err': count_uncert_ratio(nagb, nrgb),
                'nrgb': nrgb,
                'nagb': nagb}
    narratio_line += narratio_fmt % out_dict
    return narratio_line


def gather_results(sgal, target, inds, filter1=None, filter2=None,
                   narratio_dict=None, lf_line='', narratio_line=''):
    '''gather results into strings: call tpagb_lf and narratio'''

    if len(narratio_dict['sim_agb']) > 0:
        lf_line = tpagb_lf(sgal, narratio_dict, inds, filter1, filter2,
                           lf_line=lf_line)

    rgb = narratio_dict['sim_rgb']
    agb = narratio_dict['sim_agb']

    nrgb = float(len(rgb))
    nagb = float(len(agb))

    narratio_line = narratio('sim', nrgb, nagb, filter1, filter2,
                             narratio_line=narratio_line)

    return lf_line, narratio_line


def write_results(res_dict, target, filter1, filter2, outfile_loc, agb_mod, extra_str=''):
    '''
    Write results of VSFH output dict to files.

    Paramaters
    ----------
    res_dict : dict
        output of run_once keys with %s_line will be written to a file

    agb_mod, target, filter1, filter2, extra_str : strings
        file name formatting stings

    outfile_loc : string
        path to write output file

    Returns
    -------
    fdict : dictionary
        file and path to file
        ex: lf_file: <path_to_lf_file>
    '''
    narratio_header = '# target filter1 filter2 nrgb nagb ar_ratio ar_ratio_err \n'

    fdict = {}
    for key, line in res_dict.items():
        name = key.replace('_line', '')
        fname = ('_'.join(['%s' % s for s in (target, filter1, filter2, agb_mod, name)])).lower()

        fname = os.path.join(outfile_loc, '%s%s.dat' % (fname, extra_str))
        with open(fname, 'a') as fh:
            if 'narratio' in key:
                fh.write(narratio_header)
            if type(line) == str:
                line = [line]
            [fh.write('%s \n' % l) for l in line]
        fdict['%s_file' % name] = fname
    return fdict


def count_rgb_agb(filename, col1=None, col2=None, yfilter='V', regions_kw={},
                  match_param=None, filterset=0):

    mag1, mag2 = load_observation(filename, col1, col2, match_param=match_param,
                                  filterset=filterset)
    ymag = mag2
    if yfilter == 'V':
        ymag = mag1

    # as long as col_min is not None, it will use magbright and col_min etc
    # as verts, so leave mag2 and mag1 as is, and if CMD has V for yaxis,
    # just relfect that in
    regions_kw['color'] = mag1 - mag2
    gal_rgb, gal_agb = rgb_agb_regions(ymag, **regions_kw)

    if col2 is not None and 'IR' in col2:
        target, _ = parse_pipeline(filename)
        ir_mtrgb = get_trgb(target, filter2='F160W', filter1=None)
        opt_mtrgb = get_trgb(target, filter2='F814W')
        trgb_color = opt_mtrgb - ir_mtrgb
        gal_agb = get_itpagb(target, regions_kw['color'], ymag, col2,
                             off=trgb_color)

    return float(len(gal_rgb)), float(len(gal_agb))


def norm_diags(sgal, narratio_dict, inds, col1, col2, args, f1, f2, regions_kw,
             multi_plot=False, filterset=0):
    def numlabel(n, string):
        return '{} {}'.format(len(n), string)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    kw = {'alpha': 0.3, 'mec': 'none', 'rasterized': True}
    ax.plot(sgal.data[f1] - sgal.data[f2],
            sgal.data[f2], '.', label=numlabel(sgal.data[f2], 'no exclude'),
            **kw)

    ax.plot(sgal.data[f1][inds] - sgal.data[f2][inds],
            sgal.data[f2][inds], '.', label=numlabel(inds, 'sim'), **kw)

    mag1, mag2 = load_observation(args.observation, col1, col2,
                                  match_param=args.match_param,
                                  filterset=filterset)
    
    ax.plot(mag1 - mag2, mag2, '.', label=numlabel(mag1, 'data'), **kw)

    ind = narratio_dict['idx_norm']
    ax.plot(sgal.data[f1][inds][ind] - sgal.data[f2][inds][ind],
            sgal.data[f2][inds][ind], '.', label=numlabel(inds[ind], 'scaled sim'), **kw)

    ind = narratio_dict['sgal_rgb']
    ax.plot(sgal.data[f1][inds][ind] - sgal.data[f2][inds][ind],
            sgal.data[f2][inds][ind], '.', label=numlabel(inds[ind], 'sim rgb'), **kw)

    ax.set_ylim(mag2[mag2<40.].max() + 0.2, mag2[mag2<40.].min() - 0.2)
    ax.set_xlim(-0.5, np.max(mag1 - mag2) + 0.1)

    plt.legend(loc='best', numpoints=1)
    ax.set_xlabel((r'${}-{}$'.format(f1, f2)).replace('_', '\_'))
    ax.set_ylabel((r'${}$'.format(f2)).replace('_', '\_'))
    plotting.add_lines_to_plot(ax, lf=False, **regions_kw)
    plt.savefig('{0:s}_diag{1:s}'.format(os.path.splitext(sgal.name)[0], EXT))
    plt.close()
    if multi_plot:
        ind = list(set(np.nonzero(sgal.data[f1][inds] < 30)[0]) &
                   set(np.nonzero(sgal.data[f2][inds] < 30)[0]) &
                   set(narratio_dict['idx_norm']))

        zdict = {'stage': [(0, 9), 10, (1, 8)],
                 'logAge': [(6, 10.1), 20, (6, 10.1)],
                 'm_ini':  [(0.9, 8.), 10, (0.9, 8.)],
                 '[M/H]': [(-2, 2), 10, (-2, 2)],
                 'C/O': [(0.48, 4), 10, (0.48, 4)],
                 'logML': [(-11, -4), 10, (-11, -4)]}

        fig, (axs) = plt.subplots(ncols=4, nrows=2, sharex=True,
                                  sharey=True, figsize=(16, 12))

        for ax in axs[:, 0]:
            ax.plot(mag1 - mag2, mag2, '.', color='k')
            ax.set_xlabel((r'${}-{}$'.format(f1,f2)).replace('_', '\_'))
            ax.set_ylabel((r'${}$'.format(f2)).replace('_', '\_'))
        xlim = (regions_kw['col_min'], ax.get_xlim()[1])
        ylim = (regions_kw['mag_faint'], ax.get_ylim()[0])

        axs = axs[:, 1:].flatten()
        for i, (zcol, dat) in enumerate(zdict.items()):
            ax = axs[i]
            clim, bins, vlims = dat
            # data model CMD plot
            ax = sgal.color_by_arg(sgal.data[f1][inds] - sgal.data[f2][inds],
                                   sgal.data[f2][inds],
                                   zcol, ax=ax, clim=clim, xlim=xlim, bins=bins,
                                   slice_inds=ind,
                                   skw={'vmin': vlims[0], 'vmax': vlims[1]})
            ax = plotting.add_lines_to_plot(ax, lf=False, **regions_kw)

        ax.set_ylim(ylim)

        plt.savefig(sgal.name.replace('.dat', '_diagcmd{}'.format(EXT)))
        plt.close()
    return

description = """Scale trilegal catalog to a CMD region of that data.

To define the CMD region, set colorlimits (optional) and mag limits.

For the mag limits either:
   a) directly set maglimits
   b) set the trgb (or if ANGST, will try to find it), trgboffset, and trgbexclude.
   c) set fake_file to a matchfake file and a completeness to a fraction
      completeness and it will choose the fainter between the two.
   d) you can also include the match param file to apply exclude regions
      (only makes sense if the filters are the same in calcsfh and here)
"""

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter,
                                     fromfile_prefix_chars='@')

    parser.add_argument('-a', '--ast_cor', action='store_true',
                        help='use ast corrected mags already in trilegal catalog')

    parser.add_argument('-b', '--comp_frac', type=float, default=None,
                        help='use the completeness fraction for the lower maglimit (need AST file)')

    parser.add_argument('-c', '--colorlimits', type=str, default=None,
                        help=('comma separated color min, color max '
                              'or color min and color offset '
                              '(color max = color min + color offset) '
                              'for normalization region'))

    parser.add_argument('-d', '--diag', action='store_true',
                        help='make cmd diagnostic plot of the normalization')

    parser.add_argument('-e', '--trgbexclude', type=float, default=0.1,
                        help='region around trgb mag to also exclude')

    parser.add_argument('-f', '--fake', type=str, help='fake file name')

    parser.add_argument('-g', '--trgboffset', type=float, default=1.,
                        help='trgb offset, mags below trgb for lower maglimit')

    parser.add_argument('-m', '--maglimits', type=str, default=None,
                        help='comma separated faint and bright yaxis mag limits')

    parser.add_argument('-n', '--norm', type=float, default=None,
                        help='override finding normalization constant with this fraction')

    parser.add_argument('-o', '--output', type=str, default=None,
                        help='create a trilegal catalog with only normalized selection')

    parser.add_argument('-q', '--colnames', type=str,
                        help='comma separated V,I names in observation data')

    parser.add_argument('--filters', type=str, default='V,I',
                        help='comma separated V,I names in trilegal catalog')

    parser.add_argument('-s', '--filterset', type=int, default=0,
                        help='if 2 filters, and the fake file has 4, provide which filters to use 0: first two or 1: second two')

    parser.add_argument('-t', '--trgb', type=float, default=None,
                        help='trgb mag (will not attempt to find it)')

    parser.add_argument('-v', '--Av', type=float, default=0.,
                        help='visual extinction')

    parser.add_argument('-y', '--yfilter', type=str, default='I',
                        help='V or I filter to use as y axis of CMD [V untested!]')

    parser.add_argument('-z', '--match_param', type=str, default=None,
                        help='overplot exclude gates from calcsfh parameter file')

    parser.add_argument('--observation', type=str,
                        help='photometry to normalize against')

    parser.add_argument('--simpop', help='trilegal catalog to normalize')

    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    # set up logging
    outfile_loc = os.path.split(args.simpop[0])[0]
    logfile = os.path.join(outfile_loc, 'normalize.log')
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.debug('command: {0!s}'.format(argv))

    # parse the CMD region
    regions_kw = parse_regions(args)
    args.target, _ = parse_pipeline(args.simpop)
    # grab data nrgb and nagb and get ready to write to narratio file
    try:
        col1, col2 = args.colnames.split(',')
    except:
        col1 = None
        col2 = None

    obs_nrgbs, obs_nagbs = count_rgb_agb(args.observation, col1=col1, col2=col2,
                                         yfilter=args.yfilter,
                                         regions_kw=regions_kw,
                                         match_param=args.match_param,
                                         filterset=args.filterset)

    filter1, filter2 = args.filters.split(',')
    narratio_line = narratio('data', obs_nrgbs, obs_nagbs, filter1, filter2)

    # allow for ast corrections instead of trilegal catalog
    lf_line = ''
    extra_str = ''
    if args.ast_cor:
        extra_str += '_ast_cor'
        f1 = '{}_cor'.format(filter1)
        f2 = '{}_cor'.format(filter2)
    else:
        f1 = filter1
        f2 = filter2

    kws = {'filter1': f1, 'filter2': f2}
    tricat = args.simpop
    logger.debug('normalizing: {}'.format(tricat))

    sgal, narratio_dict, inds = \
        do_normalization(yfilter=args.yfilter, Av=args.Av, tricat=tricat,
                         nrgbs=obs_nrgbs, regions_kw=regions_kw, tol=1000,
                         norm=args.norm, match_param=args.match_param,
                         **kws)

    if args.diag:
        args.multi = False  # do I need another arg flag?!
        norm_diags(sgal, narratio_dict, inds, col1, col2, args, f1, f2,
                   regions_kw, multi_plot=args.multi, filterset=args.filterset)

    if args.output is not None:
        sgal.write_data(args.output, overwrite=True,
                        slice_inds=narratio_dict['idx_norm'])
    else:
        lf_line, narratio_line = gather_results(sgal, args.target, inds,
                                                narratio_dict=narratio_dict,
                                                narratio_line=narratio_line,
                                                lf_line=lf_line, **kws)


        result_dict = {'lf_line': lf_line, 'narratio_line': narratio_line}
        #result_dict['contam_line'] = contamination_by_phases(sgal, sgal_rgb,
        #                                                     sgal_agb, filter2)

        # write the output files
        agb_mod = 'agb model'
        write_results(result_dict, args.target, f1, f2, outfile_loc, agb_mod,
                      extra_str=extra_str)

    return


if __name__ == "__main__":
    sys.exit(main())
