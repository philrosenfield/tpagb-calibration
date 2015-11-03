"""
Utilities to analyze trilegal output catalogs of TPAGB models
"""
import argparse
import logging
import numpy as np
import os
import sys

import ResolvedStellarPops as rsp
from ..TPAGBparams import EXT
from .analyze import get_itpagb, parse_regions
from ..pop_synth.stellar_pops import normalize_simulation, rgb_agb_regions, limiting_mag, exclude_gate_inds
from ..sfhs.star_formation_histories import StarFormationHistories
from ..fileio import load_observation, load_table
from ..plotting import plotting

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from argparse import RawTextHelpFormatter

from ..TPAGBparams import snap_src, matchfake_loc

def do_normalization(yfilter=None, filter1=None, filter2=None, ast_cor=False,
                     sgal=None, tricat=None, nrgbs=None, regions_kw={}, Av=0.,
                     match_param=None, norm=None):
    '''Do the normalization: call rgb_agb_regions and normalize_simulations.'''

    if sgal is None:
        sgal = rsp.SimGalaxy(tricat)
        if 'dav' in tricat.lower():
            print('applying dav')
            dAv = float('.'.join(sgal.name.split('dav')[1].split('.')[:2]).replace('_',''))
            sgal.data['F475W'] += sgal.apply_dAv(dAv, 'F475W', 'phat', Av=Av)
            sgal.data['F814W'] += sgal.apply_dAv(dAv, 'F814W', 'phat', Av=Av)

    if match_param is not None:
        target, (f1, f2) = rsp.asts.parse_pipeline(match_param)
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
    target = os.path.split(tricat)[1].split('_')[1]

    sgal_rgb, sgal_agb = rgb_agb_regions(ymag, **regions_kw)

    if 'IR' in filter2 in filter2 or '160' in filter2 and not '110' in filter1:
        sgal_agb = get_itpagb(target, regions_kw['color'], ymag, filter2)

    # normalization
    norm, idx_norm, sim_rgb, sim_agb = normalize_simulation(ymag, nrgbs,
                                                            sgal_rgb, sgal_agb,
                                                            norm=norm)
    tol = 1. #0.7
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
    """format a narratio_dict for a line in the LF output file"""

    header = '# {} {} '.format(filt1, filt2)
    header += 'logAge [M/H] sim_rgb sim_agb sgal_rgb sgal_agb idx_norm norm'

    if len(lf_line) == 0:
        lf_line = header
    lf_line += '\n' + '\n'.join([' '.join(['%g' % m for m in sgal.data[filt1][inds]]),
                                 ' '.join(['%g' % m for m in sgal.data[filt2][inds]]),
                                 ' '.join(['%g' % m for m in sgal.data['logAge'][inds]]),
                                 ' '.join(['%g' % m for m in sgal.data['[M/H]'][inds]]),
                                 ' '.join(['%i' % m for m in narratio_dict['sim_rgb']]),
                                 ' '.join(['%i' % m for m in narratio_dict['sim_agb']]),
                                 ' '.join(['%i' % m for m in narratio_dict['sgal_rgb']]),
                                 ' '.join(['%i' % m for m in narratio_dict['sgal_agb']]),
                                 ' '.join(['%i' % m for m in narratio_dict['idx_norm']]),
                                 '%.4f' % narratio_dict['norm']])
    return lf_line


def narratio(target, nrgb, nagb, filt1, filt2, narratio_line=''):
    """format numbers of stars for the narratio table"""
    # N agb/rgb ratio file
    narratio_fmt = '%(target)s %(filter1)s %(filter2)s %(nrgb)i %(nagb)i %(ar_ratio).3f %(ar_ratio_err).3f\n'
    try:
       ar_ratio = nagb / nrgb
    except ZeroDivisionError:
        ar_ratio = np.nan
    out_dict = {'target': target,
                'filter1': filt1,
                'filter2': filt2,
                'ar_ratio': ar_ratio,
                'ar_ratio_err': rsp.utils.count_uncert_ratio(nagb, nrgb),
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

    narratio_line = narratio(target, nrgb, nagb, filter1, filter2,
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


def count_rgb_agb(filename, col1, col2, yfilter='V', regions_kw={},
                  match_param=None):
    target, _ = rsp.asts.parse_pipeline(filename)
    mag1, mag2 = load_observation(filename, col1, col2, match_param=match_param)
    ymag = mag2
    if yfilter == 'V':
        ymag = mag1

    # as long as col_min is not None, it will use magbright and col_min etc
    # as verts, so leave mag2 and mag1 as is, and if CMD has V for yaxis,
    # just relfect that in
    regions_kw['color'] = mag1 - mag2
    gal_rgb, gal_agb = rgb_agb_regions(ymag, **regions_kw)
    if 'IR' in col2:
        gal_agb = get_itpagb(target, regions_kw['color'], ymag, col2)

    return float(len(gal_rgb)), float(len(gal_agb))


def norm_diags(sgal, narratio_dict, inds, col1, col2, args, f1, f2, regions_kw,
             multi_plot=False):
    def numlabel(n, string):
        return '{} {}'.format(len(n), string)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    kw = {'alpha': 0.3, 'mec': 'none'}
    ax.plot(sgal.data[f1] - sgal.data[f2],
            sgal.data[f2], '.', label=numlabel(sgal.data[f2], 'no exclude'),
            **kw)

    ax.plot(sgal.data[f1][inds] - sgal.data[f2][inds],
            sgal.data[f2][inds], '.', label=numlabel(inds, 'sim'), **kw)

    mag1, mag2 = load_observation(args.observation, col1, col2,
                                  match_param=args.match_param)
    ax.plot(mag1 - mag2, mag2, '.', label=numlabel(mag1, 'data'), **kw)

    ind = narratio_dict['idx_norm']
    ax.plot(sgal.data[f1][inds][ind] - sgal.data[f2][inds][ind],
            sgal.data[f2][inds][ind], '.', label=numlabel(inds[ind], 'scaled sim'), **kw)

    ind = narratio_dict['sgal_rgb']
    ax.plot(sgal.data[f1][inds][ind] - sgal.data[f2][inds][ind],
            sgal.data[f2][inds][ind], '.', label=numlabel(inds[ind], 'sim rgb'), **kw)

    ax.set_ylim(mag2.max() + 0.2, mag2.min() - 0.2)
    ax.set_xlim(-0.5, np.max(mag1 - mag2) + 0.1)

    plt.legend(loc='best', numpoints=1)
    ax.set_xlabel('{}-{}'.format(f1,f2))
    ax.set_ylabel('{}'.format(f2))
    plotting.add_lines_to_plot(ax, lf=False, **regions_kw)
    plt.savefig(sgal.name.replace('.dat', '_diag{}'.format(EXT)))
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

def main(argv):
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
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('-a', '--ast_cor', action='store_true',
                        help='use ast corrected mags already in trilegal catalog')

    parser.add_argument('-b', '--comp_frac', type=float, default=None,
                        help='completeness fraction')

    parser.add_argument('-c', '--colorlimits', type=str, default=None,
                        help=('comma separated color min, color max '
                              'or color min and color offset '
                              '(color max = color min + color offset) '
                              'for normalization region'))

    parser.add_argument('-d', '--diag', action='store_true',
                        help='make cmd diagnostic plot of the normalization')

    parser.add_argument('-e', '--trgbexclude', type=float, default=0.1,
                        help='comma separated regions around trgb to exclude')

    parser.add_argument('-g', '--trgboffset', type=float, default=1.,
                        help='trgb offset, mags below trgb')

    parser.add_argument('-m', '--maglimits', type=str, default=None,
                        help='comma separated faint and bright yaxis mag limits')

    parser.add_argument('-n', '--norm', type=float, default=None,
                        help='override finding normalization constant with this fraction')

    parser.add_argument('-o', '--output', type=str, default=None,
                        help='create a trilegal catalog with only normalized selection')

    parser.add_argument('-p', '--lfplot', action='store_true',
                        help='plot the resulting scaled lf function against data')

    parser.add_argument('-q', '--colnames', type=str, default='MAG2_ACS,MAG4_IR',
                        help='comma separated column names in observation data')

    parser.add_argument('-s', '--scolnames', type=str, default='F814W,F160W',
                        help='comma separated column names in trilegal catalog')

    parser.add_argument('-t', '--trgb', type=str, default=None,
                        help='trgb mag')

    parser.add_argument('-v', '--Av', type=float, default=0.,
                        help='visual extinction')

    parser.add_argument('-y', '--yfilter', type=str, default='I',
                        help='V or I filter to use as y axis of CMD [V untested!]')

    parser.add_argument('-z', '--match_param', type=str, default=None,
                        help='exclude gates from calcsfh parameter file')

    parser.add_argument('observation', type=str,
                        help='photometry to normalize against')

    parser.add_argument('simpop', nargs='*',
                        help='trilegal catalog(s) to normalize')

    args = parser.parse_args(argv)

    # set up logging
    outfile_loc = os.path.split(args.simpop[0])[0]
    logfile = os.path.join(outfile_loc, 'normalize.log')
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.debug('command: {}'.format(' '.join(argv)))

    # parse the CMD region
    args.target = args.simpop[0].split('_')[1]  # maybe this should be command line?
    regions_kw = parse_regions(args)

    # grab data nrgb and nagb and get ready to write to narratio file
    col1, col2 = args.colnames.split(',')
    obs_nrgbs, obs_nagbs = count_rgb_agb(args.observation, col1, col2,
                                         yfilter=args.yfilter,
                                         regions_kw=regions_kw,
                                         match_param=args.match_param)
    filter1, filter2 = args.scolnames.upper().split(',')
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

    for tricat in args.simpop:
        logger.debug('normalizing: {}'.format(tricat))

        sgal, narratio_dict, inds = \
            do_normalization(yfilter=args.yfilter, Av=args.Av, tricat=tricat,
                             nrgbs=obs_nrgbs, regions_kw=regions_kw,
                             norm=args.norm, match_param=args.match_param,
                             **kws)

        if len(inds) == 0:
            # normalization > tolorance
            continue

        lf_line, narratio_line = gather_results(sgal, args.target, inds,
                                                narratio_dict=narratio_dict,
                                                narratio_line=narratio_line,
                                                lf_line=lf_line, **kws)

        if args.diag:
            args.multi = False  # do I need another arg flag?!
            norm_diags(sgal, narratio_dict, inds, col1, col2, args, f1, f2,
                       regions_kw, multi_plot=args.multi)

        if args.output is not None:
            sgal.write_data(args.output, overwrite=True,
                            slice_inds=narratio_dict['idx_norm'])
        del sgal  # does this save time?

    result_dict = {'lf_line': lf_line, 'narratio_line': narratio_line}
    #result_dict['contam_line'] = contamination_by_phases(sgal, sgal_rgb,
    #                                                     sgal_agb, filter2)

    # write the output files
    agb_mod = tricat.split('_')[6]
    fdict = write_results(result_dict, args.target, f1, f2, outfile_loc, agb_mod,
                          extra_str=extra_str)

    if args.lfplot:
        plotting.compare_to_gal(fdict['lf_file'], args.observation,
                       narratio_file=fdict['narratio_file'], filter1=f1,
                       agb_mod=agb_mod, regions_kw=regions_kw,
                       xlims=[(19,26), (18, 25)], filter2=f2,
                       col1=col1, col2=col2, match_param=args.match_param)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
