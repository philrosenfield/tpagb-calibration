import logging
import os
import sys

import numpy as np
import ResolvedStellarPops as rsp

from astropy.table import Table
from TPAGBparams import snap_src, phat_src


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_loc = os.path.join(snap_src, 'data', 'galaxies')
match_run_loc = os.path.join(snap_src, 'match')
phat_data_loc =  os.path.join(phat_src, 'low_av', 'phot')
phat_match_run_loc =  os.path.join(phat_src, 'low_av', 'fake')

def load_phat(target):
    galname, = rsp.fileio.get_files(phat_data_loc, '*{}*match'.format(target))
    optgal = rsp.StarPop()
    optgal.data = np.genfromtxt(galname, unpack=True, names=['F475W', 'F814W'])
    return optgal, optgal

def find_phatfake(target):
    fake, = rsp.fileio.get_files(phat_match_run_loc,
                                '*{}*.matchfake'.format(target))  
    return fake, fake

def load_obs(target, optfilter1=''):
    """return in NIR and OPT galaxy as StarPop objects"""
    from astropy.io import fits
    if 'm31' in target or 'B' in target:
        optgal, nirgal = load_phat(target)
    else:
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
    if 'm31' in target or 'B' in target:
        optfake, nirfake = find_phatfake(target)
    else:
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


def load_lf_file(lf_file):
    """
    Read a strange formatted file... basically, the rows are transformed.
    
    Each data row of the file corresponds to a header key.
    
    For example, data row 1 corresponds to header key 1, data row 2 corresponds
    to header key 2 ... up to the number of header keys, and then repeats.
    So if there are 5 header keys, data row 6 corresponds to header key 1, etc.
    
    It was done this way because some header keys correspond to a float,
    some correspond to an array of variable length.
    """
    header = open(lf_file).readline().replace('#', '').split()
    ncols = len(header)

    with open(lf_file, 'r') as lf:
        lines = [l.strip() for l in lf.readlines() if not l.startswith('#')]

    lfd = {}
    for i, key in enumerate(header):
        lfd[key] = [np.array(map(rsp.utils.is_numeric, l.split()))
                    for l in lines[i::ncols]]

    return lfd


def load_observation(filename, colname1, colname2):
    if filename.endswith('fits'):
        data = Table.read(filename, format='fits')
        mag1 = data[colname1]
        mag2 = data[colname2]
    else:
        mag1, mag2 = np.loadtxt(filename)
    return mag1, mag2


