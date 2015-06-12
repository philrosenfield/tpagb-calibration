"""
Utilities to analyze trilegal output catalogs of TPAGB models
"""
import argparse
import logging
import numpy as np
import os
import sys
import time

import ResolvedStellarPops as rsp

from astropy.io import ascii
from .analyze import get_itpagb
from ..pop_synth.stellar_pops import normalize_simulation, rgb_agb_regions, limiting_mag
from ..sfhs.star_formation_histories import StarFormationHistories
from ..fileio import load_observation
from ..utils import check_astcor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def do_normalization(yfilter=None, filter1=None, filter2=None, ast_cor=False,
                     sgal=None, tricat=None, nrgbs=None, regions_kw={}, Av=0.):
    '''Do the normalization: call rgb_agb_regions and normalize_simulations.'''

    if sgal is None:
        sgal = rsp.SimGalaxy(tricat)
        if 'dav' in tricat.lower():
            print('applying dav')
            dAv = float('.'.join(sgal.name.split('dav')[1].split('.')[:2]).replace('_',''))
            sgal.data['F475W'] += sgal.apply_dAv(dAv, 'F475W', 'phat', Av=Av)
            sgal.data['F814W'] += sgal.apply_dAv(dAv, 'F814W', 'phat', Av=Av)
    
    if ast_cor:
        filter1 += '_cor'
        filter2 += '_cor'

    ymag = sgal.data[filter2]
    if yfilter == 'V':
        ymag = sgal.data[filter1]

    regions_kw['color'] = sgal.data[filter1] - sgal.data[filter2]

    # select rgb and agb regions
    sgal_rgb, x = rgb_agb_regions(ymag, **regions_kw)
    
    target = os.path.split(tricat)[1].split('_')[1]
    sgal_agb = get_itpagb(target, regions_kw['color'], ymag)
    
    # normalization
    norm, idx_norm, sim_rgb, sim_agb = normalize_simulation(ymag, nrgbs,
                                                            sgal_rgb, sgal_agb)

    norm_dict = {'norm': norm, 'sim_rgb': sim_rgb, 'sim_agb': sim_agb,
                 'sgal_rgb': sgal_rgb, 'sgal_agb': sgal_agb,
                 'idx_norm': idx_norm}

    return sgal, norm_dict


def tpagb_lf(sgal, narratio_dict, filt1, filt2, lf_line=''):
    """format a narratio_dict for a line in the LF output file"""

    header = '# {} {} '.format(filt1, filt2)
    header += 'logAge [M/H] sim_rgb sim_agb sgal_rgb sgal_agb idx_norm norm\n'
    
    if len(lf_line) == 0:
        lf_line = header
    lf_line += '\n'.join([' '.join(['%g' % m for m in sgal.data[filt1]]),
                          ' '.join(['%g' % m for m in sgal.data[filt2]]),
                          ' '.join(['%g' % m for m in sgal.data['logAge']]),
                          ' '.join(['%g' % m for m in sgal.data['[M/H]']]),
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

    out_dict = {'target': target,
                'filter1': filt1,
                'filter2': filt2,
                'ar_ratio': nagb / nrgb,
                'ar_ratio_err': rsp.utils.count_uncert_ratio(nagb, nrgb),
                'nrgb': nrgb,
                'nagb': nagb}
    narratio_line += narratio_fmt % out_dict
    return narratio_line


def gather_results(sgal, target, filter1=None, filter2=None, ast_cor=False,
                   narratio_dict=None, lf_line='', narratio_line=''):
    '''gather results into strings: call tpagb_lf and narratio'''
    filt1, filt2 = [filter1, filter2]
    if ast_cor:
        filt1, filt2 = check_astcor([filter1, filter2])

    lf_line = tpagb_lf(sgal, narratio_dict, filt1, filt2, lf_line=lf_line)

    rgb = narratio_dict['sim_rgb']
    agb = narratio_dict['sim_agb']
    
    nrgb = float(len(rgb))
    nagb = float(len(agb))
    
    narratio_line = narratio(target, nrgb, nagb, filt1, filt2,
                             narratio_line=narratio_line)

    return lf_line, narratio_line


def write_results(res_dict, target, filter1, filter2, outfile_loc, extra_str=''):
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
        fname = ('_'.join(['%s' % s for s in (target, filter1,
                                              filter2, name)])).lower()

        fname = os.path.join(outfile_loc, '%s%s.dat' % (fname, extra_str))
        with open(fname, 'a') as fh:
            if 'narratio' in key:
                fh.write(narratio_header)
            if type(line) == str:
                line = [line]
            [fh.write('%s \n' % l) for l in line]
        fdict['%s_file' % name] = fname
    return fdict


def get_trgb(target, filter2='F160W'):
    import difflib
    angst_data = rsp.angst_tables.angst_data
    if 'm31' in target.lower() or 'B' in target:
        trgb = 22.
    else:
        if not '160' in filter2:
            angst_target = difflib.get_close_matches(target.upper(),
                                                     angst_data.targets)[0].replace('-', '_')
            
            target_row = angst_data.__getattribute__(angst_target)
            key, = [k for k in target_row.keys() if ',' in k]
            trgb = target_row[key]['mTRGB']
        else:
            trgb = angst_data.get_snap_trgb_av_dmod(target.upper())[0]
    return trgb


def load_table(filename, target, optfilter1=None, opt=True):
    
    if opt:
        filt1 = optfilter1
        assert optfilter1 is not None
    else:
        filt1 = nirfilter1

    tbl = ascii.read(filename)

    ifilts = list(np.nonzero((tbl['filter1'] == filt1))[0])
    itargs = [i for i in range(len(tbl['target'])) if target.upper() in tbl['target'][i]]
    indx = list(set(ifilts) & set(itargs))

    if len(indx) == 0:
        logger.error('{}, {} not found in table'.format(target, filt1))
        sys.exit(2)

    return tbl[indx]


def parse_regions(args):    
    if not hasattr(args, 'table'):
        args.table = None
    # need the following in opt and nir
    colmin, colmax = None, None
    magfaint, magbright = None, None
    
    opt = True
    if args.yfilter == 'V':
        opt = False
    filter1, filter2 = args.scolnames.split(',')
    if args.trgb is None:
        # def filter2
        trgb = get_trgb(args.target, filter2=filter2)
    
    if args.table is not None:
        row = load_table(args.table, args.target, optfilter1=args.optfilter1,
                         opt=opt)
        if args.offset is None or row['mag_by_eye'] != 0:
            logger.debug('mags to norm to rgb are set by eye from table')
            magbright = row['magbright']
            magfaint = row['magfaint']
        else:
            magbright = trgb + trgbexclude
            if row['comp90mag2'] < trgb + args.trgboffset:
                msg = '0.9 completeness fraction'
                opt_magfaint = row['comp90mag2']
            else:
                msg = 'trgb + offset'
                magfaint = trgb + offset
            logger.debug('faint mag limit for rgb norm set to {}'.format(msg))

        colmin = row['colmin']
        colmax = row['colmax']
    else:
        if args.colorlimits is not None:
            colmin, colmax = map(float, args.colorlimits.split(','))
            if colmax < colmin:
                colmax = colmin + colmax
                logger.debug('colmax was less than colmin, assuming it is dcol, colmax is set to colmin + dcol')

        if args.maglimits is not None:
            magfaint, magbright = map(float, args.maglimits.split(','))
        
        if args.trgbexclude is not None:    
            magbright = trgb + args.trgbexclude
            magfaint = trgb + args.trgboffset
            msg = 'trgb + offset'

            if args.comp_frac is not None:
                comp_mag1, comp_mag2 = limiting_mag(args.fake_file,
                                                    args.comp_frac) 
                if comp_mag2 < trgb + args.trgboffset:
                    msg = '{} completeness fraction: {}'.format(args.comp_frac,
                                                                comp_mag2)
                    magfaint = comp_mag2
                else:
                    logger.debug('magfaint: {} comp_mag2: {} using magfaint'.format(magfaint, comp_mag2))
            logger.debug('faint mag limit for rgb norm set to {}'.format(msg))

    regions_kw = {'offset': args.trgboffset,
                  'trgb_exclude': args.trgbexclude,
                  'trgb': trgb,
                  'col_min': colmin,
                  'col_max': colmax,
                  'mag_bright': magbright,
                  'mag_faint': magfaint}
    logger.debug('regions: {}'.format(regions_kw))
    return regions_kw


def count_rgb_agb(filename, col1, col2, yfilter='V', regions_kw={}):
    target = os.path.split(filename)[1].split('_')[1]
    mag1, mag2 = load_observation(filename, col1, col2)
    ymag = mag2
    if yfilter == 'V':
        ymag = mag1
    
    # as long as col_min is not None, it will use magbright and col_min etc
    # as verts, so leave mag2 and mag1 as is, and if CMD has V for yaxis,
    # just relfect that in
    regions_kw['color'] = mag1 - mag2
    gal_rgb, _ = rgb_agb_regions(ymag, **regions_kw)
    gal_agb = get_itpagb(target, regions_kw['color'], ymag)
    
    return float(len(gal_rgb)), float(len(gal_agb))


def main(argv):
    description = ("Scale trilegal catalog to a CMD region of that data",
                 "To define the CMD region, set colorlimits (optional)",
                 "For the mag limits, either directly set maglimits or",
                 "Set the trgb (or if ANGST will try to find it)",
                 "a trgboffset (mags below trgb), and trgbexclude",
                 " (region around the trgb to not consider).",
                 "Can also set fake_file to a matchfake file and a completeness ",
                 "to a fraction completeness and it will choose the fainter between",
                 " ")
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-a', '--ast_cor', action='store_true',
                        help='use ast corrected mags already in trilegal catalog')
    
    parser.add_argument('-b', '--comp_frac', type=float, default=None,
                        help='completeness fraction for use in combo with -f')

    parser.add_argument('-c', '--colorlimits', type=str, default=None,
                        help='comma separated color min, color max')

    parser.add_argument('-d', '--directory', action='store_true',
                        help='opperate on *_???.dat files in a directory')

    parser.add_argument('-e', '--trgbexclude', type=float, default=0.1,
                        help='comma separated regions around trgb to exclude')

    parser.add_argument('-f', '--fake_file', type=str, default=None,
                        help='AST file if -b flag should match filter2 with yfilter')

    parser.add_argument('-g', '--trgboffset', type=float, default=1.,
                        help='trgb offset')

    parser.add_argument('-m', '--maglimits', type=str, default=None,
                        help='comma separated faint and bright yaxis mag limits')

    parser.add_argument('-q', '--colnames', type=str, default='MAG2_ACS,MAG4_IR',
                        help='comma separated column names in observation data')

    parser.add_argument('-p', '--lfplot', action='store_true',
                        help='plot the resulting scaled lf function against data')
    
    parser.add_argument('-r', '--table', type=str,
                        help='read colorlimits, completness mags from a prepared table')

    parser.add_argument('-s', '--scolnames', type=str, default='F814W,F160W',
                        help='comma separated column names in trilegal catalog')

    parser.add_argument('-t', '--trgb', type=str, default=None,
                        help='trgb mag')

    parser.add_argument('-v', '--Av', type=float, default=0.,
                        help='visual extinction')

    parser.add_argument('-y', '--yfilter', type=str, default='I',
                        help='V or I filter to use as y axis of CMD')

    parser.add_argument('observation', type=str,
                        help='photometry to normalize against')

    parser.add_argument('simpop', nargs='*',
                        help='trilegal catalog(s) or directory if -d flag')

    # parser: args.observation is photometry
    args = parser.parse_args(argv)

    if args.directory:
        tricats = rsp.fileio.get_files(args.simpop[0], '*_???.dat')
        outfile_loc = args.simpop[0]
        args.target = args.simpop[0]
    else:
        tricats = args.simpop
        outfile_loc = os.path.split(args.simpop[0])[0]
        args.target = args.simpop[0].split('_')[1]

    # set up logging
    logfile = os.path.join(outfile_loc, 'normalize.log')
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setLevel(logging.WARNING)
    logger.setLevel(logging.DEBUG)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.debug('command: {}'.format(' '.join(argv)))

    regions_kw = parse_regions(args)
    
    col1, col2 = args.colnames.split(',')
    filter1, filter2 = args.scolnames.split(',')

    obs_nrgbs, obs_nagbs = count_rgb_agb(args.observation, col1, col2,
                                         yfilter=args.yfilter,
                                         regions_kw=regions_kw)
    
    narratio_line = narratio('data', obs_nrgbs, obs_nagbs, filter1, filter2)

    lf_line = ''
    extra_str = ''
    if args.ast_cor:
        extra_str += '_ast_cor'
        f1 = '{}_cor'.format(filter1)
        f2 = '{}_cor'.format(filter2)
    else:
        f1 = filter1
        f2 = filter2
        
    kws = {'filter1': filter1, 'filter2': filter2, 'ast_cor': args.ast_cor}

        
    for tricat in tricats:
        logger.debug('normalizing: {}'.format(tricat))
        sgal, narratio_dict = do_normalization(yfilter=args.yfilter,
                                               tricat=tricat,
                                               nrgbs=obs_nrgbs, Av=args.Av,
                                               regions_kw=regions_kw, **kws)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            
            kw = {'alpha': 0.3, 'mec': 'none'}
            ax.plot(sgal.data[f1] - sgal.data[f2],
                    sgal.data[f2], '.', label='sim', **kw)
            
            mag1, mag2 = load_observation(args.observation, col1, col2)
            ax.plot(mag1-mag2, mag2, '.', label='data')
    
            ind = narratio_dict['idx_norm']
            ax.plot(sgal.data[f1][ind] - sgal.data[f2][ind],
                    sgal.data[f2][ind], '.', label='scaled sim', **kw)
            
            ind = narratio_dict['sgal_rgb']
            ax.plot(sgal.data[f1][ind] - sgal.data[f2][ind],
                    sgal.data[f2][ind], '.', label='sim rgb', **kw)
            
            ax.set_ylim(mag2.max() + 0.2, mag2.min() - 0.2)
            ax.set_xlim(np.min(mag1 - mag2) - 0.1, np.max(mag1 - mag2) + 0.1)
            import pdb; pbd.set_trace()
            plt.legend(loc='best', numpoints=1)
            plt.savefig(sgal.name + 'diag.png')
        except:
            pass
        
        lf_line, narratio_line = gather_results(sgal, args.target,
                                                narratio_dict=narratio_dict,                                                
                                                narratio_line=narratio_line,
                                                lf_line=lf_line, **kws)
        # does this save time?
        del sgal
    
    result_dict = {'lf_line': lf_line, 'narratio_line': narratio_line}
    #result_dict['contam_line'] = contamination_by_phases(sgal, sgal_rgb,
    #                                                     sgal_agb, filter2)
    
    # write the output files
    write_results(result_dict, args.target, filter1, filter2, outfile_loc,
                  extra_str=extra_str)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
