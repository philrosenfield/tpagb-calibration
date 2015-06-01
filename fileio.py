import logging
import os
import sys

import numpy as np
import ResolvedStellarPops as rsp

from astropy.table import Table
from TPAGBparams import snap_src, phat_src

from pop_synth.stellar_pops import limiting_mag

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
            if optfilter1 != '':
                optfake, = [o for o in optfake if optfilter1 in o]
            else:
                optfake = optfake[0]
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

def load_photometry(lf_file, observation, filter1='F814W_cor',
                     filter2='F160W_cor', col1='MAG2_ACS',
                     col2='MAG4_IR', yfilter='I', flatten=True,
                     optfake=None, comp_frac=None, nirfake=None):
    
    # load models
    lf_dict = load_lf_file(lf_file)
    
    # idx_norm are the normalized indices of the simulation set to match RGB
    idx_norm = lf_dict['idx_norm']
    
    # nmodels corresponds to the number of trilegal simulations that are in
    # the lf_file.
    nmodels = len(idx_norm)
    
    # take all the scaled cmds in the file together and make an average hess

    smag1 = np.array([lf_dict[filter1][i][idx_norm[i]]
                            for i in range(nmodels)])
    smag2 = np.array([lf_dict[filter2][i][idx_norm[i]]
                            for i in range(nmodels)])
    if flatten:
        smag1 = np.concatenate(smag1)
        smag2 = np.concatenate(smag2)
    
    # load observation
    mag1, mag2 = load_observation(observation, col1, col2)

    # for opt_nir_matched data, take the obs limits from the data
    inds, = np.nonzero((smag1 <= mag1.max()) & (smag2 <= mag2.max()) &
                       (smag1 >= mag1.min()) & (smag2 >= mag2.min()))
    if comp_frac is not None:
        if optfake is None:
            target = os.path.split(lf_file)[1].split('_')[0]
            optfake, nirfake = find_fakes(target)
        _, comp1 = limiting_mag(optfake, comp_frac)
        _, comp2 = limiting_mag(nirfake, comp_frac)
        sinds, = np.nonzero((smag2 <= comp2))# & (smag1 <= comp1))
        inds = list(set(sinds) & set(inds))
        oinds, = np.nonzero((mag2 <= comp2))# & (mag1 <= comp1))
        mag1 = mag1[oinds]
        mag2 = mag2[oinds]
        
    color = mag1 - mag2

    smag1 = smag1[inds]
    smag2 = smag2[inds]

    scolor = smag1 - smag2

    # set the yaxis
    symag = smag2
    ymag = mag2
    if yfilter.upper() != 'I':
        symag = smag1
        ymag = mag1
    
    return color, ymag, scolor, symag, nmodels

