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

from astropy.io import ascii
from ..pop_synth.stellar_pops import normalize_simulation, rgb_agb_regions, limiting_mag
from ..sfhs.star_formation_histories import StarFormationHistories
from ..fileio import load_obs, find_fakes, get_files
from ..utils import astronomy_utils
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from ..TPAGBparams import snap_src, matchfake_loc
from .. import angst_tables
angst_data = angst_tables.angst_data

__all__ = ['get_itpagb', 'parse_regions']


def get_trgb(target, filter2='F160W', filter1=None):
    import difflib
    angst_data = angst_tables.angst_data
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
                    logger.warning('More than one ANGST entry for trgb, need another filter to find trgb, assuming',target_row[key[0]])
                    logger.warning('Only worry if dmag={} is a big deal'.format(np.abs(target_row[key[0]]['mTRGB']-target_row[key[1]]['mTRGB'])))
            trgb = target_row[key[0]]['mTRGB']
        else:
            trgb = angst_data.get_snap_trgb_av_dmod(target.upper())[0]
    return trgb


def parse_regions(args):
    """
    """
    # need the following in opt and nir
    colmin, colmax = None, None
    magfaint, magbright = None, None
    filter1, filter2 = args.filters.upper().split(',')
    logger.info('filter1: {} filter2: {}'.format(filter1,filter2))

    opt = True
    if args.yfilter == 'V':
        opt = False

    trgb = args.trgb

    if args.colorlimits is not None:
        colmin, colmax = np.array(args.colorlimits.split(','), dtype=float)
        if colmax < colmin:
            colmax = colmin + colmax
            logger.info('colmax was less than colmin, assuming it is dcol, colmax is set to colmin + dcol')

    if args.maglimits is not None:
        magfaint, magbright = np.array(args.maglimits.split(','), dtype=float)
        msg = 'maglimits'
    else:
        magbright = trgb + args.trgbexclude
        magfaint = trgb + args.trgboffset
        msg = 'trgb + offset'

    if args.comp_frac is not None:
        fake = args.fake
        if fake is not None:
            logger.info('using fake: {}'.format(fake))
            _, comp_mag2 = limiting_mag(fake, args.comp_frac)
            #comp_mag1, comp_mag2 = limiting_mag(args.fake_file,
            #                                    args.comp_frac)
            if comp_mag2 < trgb + args.trgboffset:
                msg = '{} completeness fraction: {}'.format(args.comp_frac,
                                                            comp_mag2)
                magfaint = comp_mag2
            else:
                logger.info('magfaint: {} comp_mag2: {} using magfaint'.format(magfaint, comp_mag2))
        else:
            logger.warning('comp_frac requested, no fake file passed')

    logger.info('faint mag limit for rgb norm set to {}'.format(msg))

    regions_kw = {'offset': args.trgboffset,
                  'trgb_exclude': args.trgbexclude,
                  'trgb': trgb,
                  'col_min': colmin,
                  'col_max': colmax,
                  'mag_bright': magbright,
                  'mag_faint': magfaint}

    logger.info('regions: {}'.format(regions_kw))

    return regions_kw


def tpagb_rheb_line(mag, b=-7, m=-9.03226, dmod=0., Av=0.0, off=0.0):
    """
    Default values found in this code using UGC5139 and NGC3741
    b=6.21106, m=-8.97165
    by eye:
    b=1.17303, m=-5.20269
    median:
    b=6.653127, m=-9.03226
    set dmod and Av to 0 for absmag
    adjusted from median by eye:
    b=7.453127, m=-9.03226
    with off = f814w-f160w trgb
    b=-7, m=-9.03226
    """
    ah = 0.20443
    ai = 0.60559
    c = (ah + m * (ah - ai))
    return (mag - b - dmod - c * Av) / m + off


def get_itpagb(target, color, mag, col, blue_cut=-99, absmag=False,
               mtrgb=None, dmod=0.0, Av=0.0, off=0):
    # careful! get_snap assumes F160W
    if '160' in col or '110' in col or 'IR' in col:
        if mtrgb is None:
            print('must pass mtrgb')
        cs = tpagb_rheb_line(mag, dmod=dmod, Av=Av, off=off)
        redward_of_rheb, = np.nonzero(color > cs)
        blueward_of_rheb, = np.nonzero(color < cs)
    else:
        logger.warning('Not using TP-AGB RHeB line')
        if mtrgb is None:
            mtrgb, Av, dmod = angst_data.get_tab5_trgb_av_dmod(target.upper(),
                                                               filters=col)
            if absmag:
                mtrgb = astronomy_utils.mag2Mag(mtrgb, col, 'acs_wfc',
                                                    dmod=dmod, Av=Av)
        redward_of_rheb = np.arange(len(color))
        blueward_of_rheb = np.arange(len(color))

    redward_of_bluecut, = np.nonzero(color > blue_cut)
    brighter_than_trgb, = np.nonzero(mag < mtrgb)
    itpagb = list(set(redward_of_rheb) & set(brighter_than_trgb))
    irheb = list(set(brighter_than_trgb) & set(redward_of_bluecut) & set(blueward_of_rheb))
    return itpagb #, irheb


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
