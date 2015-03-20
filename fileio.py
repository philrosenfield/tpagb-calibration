import logging
import os
import sys

import ResolvedStellarPops as rsp
from TPAGBparams import snap_src

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_loc = os.path.join(snap_src, 'data', 'galaxies')
match_run_loc = os.path.join(snap_src, 'match')

def load_obs(target, optfilter1=''):
    """return in NIR and OPT galaxy as StarPop objects"""
    from astropy.io import fits
    nirgalname, = rsp.fileio.get_files(data_loc,
                                       '*{}*fits'.format(target.upper()))

    optgalname, = rsp.fileio.get_files(data_loc,
                                       ('*{}*{}*fits'.format(target, optfilter1).lower()))

    nirgal = rsp.StarPop()
    nirgal.data = fits.getdata(nirgalname)

    optgal = rsp.StarPop()
    optgal.data = fits.getdata(optgalname)
    return optgal, nirgal

def find_fakes(target, optfilter1=''):
    """return matchfake filenames"""
    search_str = '*{}*.matchfake'.format(target.upper())
    fakes = rsp.fileio.get_files(data_loc, search_str)

    nirfake, = [f for f in fakes if 'IR' in f]
    optfake = [f for f in fakes if not 'IR' in f]

    if len(optfake) > 1:
        optfake, = [o for o in optfake if optfilter1 in o]
    elif len(optfake) == 1:
        optfake, = optfake

    return optfake, nirfake

def find_match_param(target, optfilter1=''):
    """return matchparam filename"""
    search_str = '*{}*.param'.format(optfilter1)
    loc = os.path.join(match_run_loc, target)

    if not os.path.isdir(loc):
        logger.error('{} directory not found!'.format(loc))
        sys.exit(2)

    try:
        mparam, = rsp.fileio.get_files(loc, search_str)
    except ValueError:
        if optfilter1 == '':
            raise ValueError, 'Need to pass optfilter1'
        else:
            raise ValueError

    return mparam
