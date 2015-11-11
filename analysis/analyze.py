"""
Utilities to analyze trilegal output catalogs of TPAGB models
"""
import argparse
import logging
import numpy as np
import os
import sys
import time

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt
import ResolvedStellarPops as rsp

from astropy.io import ascii
from ..pop_synth.stellar_pops import normalize_simulation, rgb_agb_regions, limiting_mag
from ..sfhs.star_formation_histories import StarFormationHistories
from ..fileio import load_obs, find_fakes

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


from ..TPAGBparams import snap_src, matchfake_loc
angst_data = rsp.angst_tables.angst_table.AngstTables()

__all__ = ['get_itpagb', 'parse_regions']


def get_trgb(target, filter2='F160W', filter1=None):
    import difflib
    angst_data = rsp.angst_tables.angst_data
    if 'm31' in target.lower() or 'B' in target:
        trgb = 22.
    else:
        if not '160' in filter2:
            angst_target = difflib.get_close_matches(target.upper(),
                                                     angst_data.targets)[0].replace('-', '_')

            target_row = angst_data.__getattribute__(angst_target)
            key = [k for k in target_row.keys() if ',' in k]
            if len(key) > 1:
                if filter1 is not None:
                    key = [k for k in key if filter1 in k]
                else:
                    logger.error('Need another filter to find trgb')
            trgb = target_row[key[0]]['mTRGB']
        else:
            trgb = angst_data.get_snap_trgb_av_dmod(target.upper())[0]
    return trgb


def parse_regions(args):
    """
    """
    if not hasattr(args, 'table'):
        args.table = None
    # need the following in opt and nir
    colmin, colmax = None, None
    magfaint, magbright = None, None
    filter1, filter2 = args.scolnames.upper().split(',')
    logger.info('filter1: {} filter2: {}'.format(filter1,filter2))

    opt = True
    if args.yfilter == 'V':
        opt = False

    if args.trgb is None:
        try:
            trgb = get_trgb(args.target, filter2=filter2, filter1=filter1)
        except:
            trgb = np.nan

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
        else:
            magbright = trgb + args.trgbexclude
            magfaint = trgb + args.trgboffset
            msg = 'trgb + offset'

            if args.comp_frac is not None:
                search_str = '*{}*.matchfake'.format(args.target.upper())
                fakes = rsp.fileio.get_files(matchfake_loc, search_str)
                logger.info('found fakes {}'.format(fakes))
                try:
                    fake = [f for f in fakes if filter1 in f][0]
                except:
                    fake = [f for f in fakes if filter2 in f][0]
                logger.info('using fake: {}'.format(fake))
                _, comp_mag2 = limiting_mag(fake, args.comp_frac)
                #comp_mag1, comp_mag2 = limiting_mag(args.fake_file,
                #                                    args.comp_frac)
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


def tpagb_rheb_line(mag, b=6.653127, m=-9.03226, dmod=0., Av=0.0):
    """
    Default values found in this code using UGC5139 and NGC3741
    b=6.21106, m=-8.97165
    by eye:
    b=1.17303, m=-5.20269
    median:
    b=6.653127, m=-9.03226
    set dmod and Av to 0 for absmag
    """
    ah = 0.20443
    ai = 0.60559
    c = (ah + m * (ah - ai))
    return (mag - b - dmod - c * Av) / m


def get_itpagb(target, color, mag, col, blue_cut=-99, absmag=False,
               mtrgb=None, dmod=0.0, Av=0.0):
    # careful! get_snap assumes F160W
    if '160' in col or '110' in col or 'IR' in col:
        if mtrgb is None:
            try:
                mtrgb, Av, dmod = angst_data.get_snap_trgb_av_dmod(target.upper())
            except:
                logger.error('Target not found: get_snap_trgb_av_dmod {}'.format(target))
                return [np.nan] * 2
            if absmag:
                mtrgb =  rsp.astronomy_utils.mag2Mag(mtrgb, 'F160W', 'wfc3ir',
                                                     dmod=dmod, Av=Av)
                dmod = 0.
                Av = 0.
        redward_of_rheb, = np.nonzero(color > tpagb_rheb_line(mag,
                                                              dmod=dmod, Av=Av))
        blueward_of_rheb, = np.nonzero(color < tpagb_rheb_line(mag,
                                                               dmod=dmod, Av=Av))

    else:
        logger.warning('Not using TP-AGB RHeB line: {}'.format(col))
        if mtrgb is None:
            mtrgb, Av, dmod = angst_data.get_tab5_trgb_av_dmod(target.upper(),
                                                               filters=col)
            if absmag:
                mtrgb = rsp.astronomy_utils.mag2Mag(mtrgb, col, 'acs_wfc',
                                                    dmod=dmod, Av=Av)
        redward_of_rheb = np.arange(len(color))
        blueward_of_rheb = np.arange(len(color))

    redward_of_bluecut, = np.nonzero(color > blue_cut)
    brighter_than_trgb, = np.nonzero(mag < mtrgb)
    itpagb = list(set(redward_of_rheb) & set(brighter_than_trgb))
    irheb = list(set(brighter_than_trgb) & set(redward_of_bluecut) & set(blueward_of_rheb))
    return itpagb#, irheb


def main(argv):
    """main function of analyze"""
    description="Run analysis routines, so far just match_stats."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='does nothing')

    parser.add_argument('-f', '--overwrite', action='store_true',
                        help='does nothing.')

    parser.add_argument('hmc_file', type=str,
                        help='MATCH HybridMC file')

    parser.add_argument('cmd_file', type=str,
                        help='MATCH SFH file: must have the format target_filter1_filter2.extensions')

    args = parser.parse_args(argv)

    match.likelihood.match_stats(args.hmc_file, args.cmd_file, dry_run=False)


if __name__ == "__main__":
    main(sys.argv[1:])
